import math
from typing import List, Union
import tvm
from tvm import relax, tir, te, topi
import tvm.relax
from tvm.ir.module import IRModule
from tvm.relax.analysis import estimate_memory_usage
from tvm.relay import GlobalVar
from tvm.target.target import Target
import tvm.testing
from tvm.script.parser import relax as R, tir as T, ir as I
import pytest
import numpy as np
from tvm.relax.expr_functor import mutator, PyExprMutator
from tvm.relax.transform.tuning_api import Trace
from tvm.tir.schedule.schedule import Schedule

from tvm.relax.dpl.pattern import is_op, wildcard
import torch
import tvm.dlight as dl

from tvm.relax import _ffi_api


def attention_func():
    pass


def attention_backward_func():
    pass


class BlockBuilder(tvm.relax.BlockBuilder):
    # attributes defined from c++ side
    _saved_tensors: List[tvm.relax.Var]
    _saved_non_tensors: List

    def save_for_backward(*args: tvm.relax.Var):
        assert all(isinstance(arg, tvm.relax.Var) for arg in args)
        pass

    def save_non_tensor_for_backward(*args):
        pass

    def call_tir(*args):
        pass

    def emit_tir(*args):
        pass

    @property
    def saved_tensors(self):
        # handle var mapping from the original function to the gradient function
        return _ffi_api.BlockBuilderSavedTensors(self)

    @property
    def saved_non_tensor(self):
        # handle possible tir var mapping from the original function to the gradient function
        return _ffi_api.BlockBuilderSavedTensors(self)


class Function:
    @staticmethod
    def forward(ctx: BlockBuilder, q: relax.Expr, k: relax.Expr, v: relax.Expr, scale: relax.Constant) -> Union[relax.Var, List[relax.Var]]:
        res = ctx.emit_te(attention_func, q, k, v, scale)
        L = ctx.emit(res[1])
        ctx.save_non_tensor_for_backward(scale)
        ctx.save_for_backward(q, k, v, L)
        return ctx.emit(res[0])

    @staticmethod
    def backward(ctx: BlockBuilder, dO: relax.Expr) -> Union[List[relax.Expr], relax.Expr]:
        q, k, v, L = ctx.saved_tensors[0]
        scale = ctx.saved_non_tensor[0]

        res = ctx.emit_te(attention_backward_func, dO, q, k, v, L, scale)
        # require returning relax Vars?
        dQ = ctx.emit(res[0])
        dK = ctx.emit(res[1])
        dV = ctx.emit(res[2])

        return dQ, dK, dV, None

    @classmethod
    def apply(cls, *args, **kwargs) -> Union[relax.Var, List[relax.Var]]:
        class_name = cls.__module__ + "." + cls.__name__
        register_op_name = "tvm.relax.training.register_grad." + class_name
        register_func_name = register_op_name + ".backward"
        tvm.register_func(register_func_name, cls.backward)

        ctx = BlockBuilder.current()
        args = [relax.start_operator(register_op_name, arg) for arg in args]
        kwargs = {key: relax.start_operator(register_op_name, arg) for key, arg in args}
        res = cls.forward(ctx, *args, **kwargs)
        if isinstance(res, (list, tuple)):
            return [ctx.emit(relax.end_operator(register_op_name, arg)) for arg in res]
        else:
            assert isinstance(res, relax.Var)
            return ctx.emit(relax.end_operator(register_op_name, res))
