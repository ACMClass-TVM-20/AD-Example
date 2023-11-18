import os
import sys
from typing import List
from time import monotonic

import tvm
from tvm import relax, tir, te, topi
from tvm.dlight.gpu import fallback
from tvm.ir.module import IRModule
from tvm.relax.analysis import estimate_memory_usage
from tvm.relax.block_builder import BlockBuilder
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

import sys
import os


# fmt: off
@I.ir_module
class Module:
    @T.prim_func
    def compute(A: T.Buffer((T.int64(512), T.int64(4096)), "float16"), O: T.Buffer((T.int64(512), T.int64(4096)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        for i0, i1 in T.grid(T.int64(512), T.int64(4096)):
            with T.block("O"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                O[v_i0, v_i1] = T.Cast("float32", A[v_i0, v_i1])
# fmt: on

# target, dev = tvm.target.Target("nvidia/nvidia-a100"), tvm.cuda()
target, dev = tvm.target.Target("apple/m2-gpu"), tvm.metal()

cur_path = os.path.dirname(os.path.abspath(__file__))
before_path = os.path.join(cur_path, "before.py")
after_path = os.path.join(cur_path, "after.py")
suffix_map = {
    "cuda": ".cu",
    "metal": ".mtl",
    "cpu": ".ll",
}
dump_path = os.path.join(cur_path, "build" + suffix_map[target.kind.default_keys[0]])
ex_path = os.path.join(cur_path, "build" + suffix_map[target.kind.default_keys[0]] + ".so")
cubin_path = os.path.join(cur_path, "build.cubin")

sch = Schedule(Module["compute"])
blk = sch.get_block("O")
v0, v1 = sch.get_loops(blk)
v00, v01, v02, v03 = sch.split(v0, [8, 2, 8, None])
v10, v11, v12, v13 = sch.split(v1, [8, 1, 8, None])
sch.reorder(v00, v10, v01, v11, v02, v12, v03, v13)
sch.bind(v00, "blockIdx.y")
sch.bind(v10, "blockIdx.x")
sch.bind(v01, "vthread.y")
sch.bind(v11, "vthread.x")
sch.bind(v02, "threadIdx.y")
sch.bind(v12, "threadIdx.x")

sch.mod.show()
ex = tvm.build(sch.mod, target=target)

print(ex.imported_modules[0].get_source())
