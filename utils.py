from __future__ import annotations
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm import topi, relax, te
from tvm.script import tir as T
from tvm.script import relax as R


@relax.expr_functor.mutator
class LowerToTensorIR(relax.PyExprMutator):
    def __init__(self, mod: IRModule, op_map) -> None:
        super().__init__()
        self.mod_ = mod
        self.op_map = {
            tvm.ir.Op.get(k): v for k, v in op_map.items()
        }


    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)

        if call.op in self.op_map:
            return self.op_map[call.op](self.builder_, call)
        return call

    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            updated_func = self.visit_expr(func)
            self.builder_.update_func(global_var, updated_func)

        return self.builder_.get()


def map_dense(bb, call):
    x, w = call.args
    return bb.call_te(topi.nn.dense, x, w)

def map_add(bb, call):
    a, b = call.args
    return bb.call_te(topi.add, a, b)

def map_relu(bb, call):
    return bb.call_te(topi.nn.relu, call.args[0])

def map_matmul(bb, call):
    return bb.call_te(topi.matmul, call.args[0], call.args[1])

def map_softmax(bb, call):
    return bb.call_te(topi.nn.softmax, call.args[0])

@tvm.te.tag_scope(tag="crossent_output")
def crossent(x, y):
    i = te.reduce_axis((0, 10), name="i")
    return te.compute(shape=(1,), fcompute=lambda _: te.sum(-y[0, i] * te.log(x[0, i]), axis=i), name="crossent")

def map_crossent(bb, call):
    return bb.call_te(crossent, call.args[0], call.args[1])

op_map = {
  "relax.nn.dense": map_dense,
  "relax.add": map_add,
  "relax.nn.relu": map_relu,
  "relax.matmul": map_matmul,
  "relax.nn.softmax": map_softmax,
  "relax.nn.crossent": map_crossent
}

@tvm.ir.transform.module_pass(opt_level=0, name="LowerToTensorIR")
class LowerToTensorIRPass:
    """The wrapper for the LowerTensorIR pass."""
    def transform_module(self, mod, ctx):
        return LowerToTensorIR(mod, op_map).transform()