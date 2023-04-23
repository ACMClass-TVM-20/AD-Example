import tvm
from tvm import relax
from tvm._ffi.registry import get_global_func
from tvm.meta_schedule.runner.runner import Runner
from tvm.meta_schedule.runner.utils import alloc_argument_common
from tvm.relax.block_builder import BlockBuilder
from tvm.runtime.ndarray import NDArray
from tvm import te, topi, meta_schedule as ms
import numpy as np
from tvm.script import tir as T, ir as I, relax as R
from tvm.tir import Schedule


def f(x, y):
    pass


def f1(grad, x, y): # df/dx
    pass


def f2(grad, x, y): # df/dy
    pass

---> current method

@tvm.script.ir_module
class InputModule:
    @R.function
    def main_adjoint(x: R.Tensor((1, 1, 16, 16), "float32")):
        R.func_attr({"checkpoints": [(z1, z2), (z3, z4)]})
        with R.dataflow():
            z1 = f(x, x)
            z2 = f(z1, x)
            z3 = f(z2, x)
            z4 = f(z3, x)
            z5 = f(z4, x)
            z5_adjoint = R.ones_like(z5)
            z4_adjoint = f1(z5_adjoint, z4, x)
            z3_adjoint = f1(z4_adjoint, z3, x)
            z2_adjoint = f1(z3_adjoint, z2, x)
            z1_adjoint = f1(z2_adjoint, z1, x)
            x_adjoint = (
                f2(z5_adjoint, z4, x)
                + f2(z4_adjoint, z3, x)
                + f2(z3_adjoint, z2, x)
                + f2(z2_adjoint, z1, x)
                + f2(z1_adjoint, x, x)
                + f2(z1_adjoint, x, x)
            )
            R.output(z5, x_adjoint)
        return z5, x_adjoint

----> Restore old method: emit together all backward bindings of one op

@tvm.script.ir_module
class InputModule:
    @R.function
    def main_adjoint(x: R.Tensor((1, 1, 16, 16), "float32")):
        R.func_attr({"checkpoints": [(z1, z2), (z3, z4)]})
        with R.dataflow():
            z1 = f(x, x)
            z2 = f(z1, x)
            z3 = f(z2, x)
            z4 = f(z3, x)
            z5 = f(z4, x)
            z5_adjoint = R.ones_like(z5)

            z4_adjoint = f1(z5_adjoint, z4, x)
            x_adjoint_1 = f2(z5_adjoint, z4, x)

            z3_adjoint = f1(z4_adjoint, z3, x)
            x_adjoint_2_0 = f2(z4_adjoint, z3, x)
            x_adjoint_2 = x_adjoint_1 + x_adjoint_2_0

            z2_adjoint = f1(z3_adjoint, z2, x)
            x_adjoint_3_0 = f2(z3_adjoint, z2, x)
            x_adjoint_3 = x_adjoint_2 + x_adjoint_3_0

            z1_adjoint = f1(z2_adjoint, z1, x)
            x_adjoint_4_0 = f2(z2_adjoint, z1, x)
            x_adjoint_4 = x_adjoint_3 + x_adjoint_4_0

            x_adjoint_5_0 = f1(z1_adjoint, x, x)
            x_adjoint_5 = x_adjoint_4 + x_adjoint_5_0
            x_adjoint_6_0 = f2(z1_adjoint, x, x)
            x_adjoint_6 = x_adjoint_5 + x_adjoint_6_0
            R.output(z5, x_adjoint_6)
        return z5, x_adjoint_6

Special case: tuplegetitem

suppose a = (b, c)
tmp = a[0]

--> gradient

a_adjoint[0] += tmp_adjoint

--> expand to current relax grammar

b_adjoint = a_adjoint[0]
c_adjoint = a_adjoint[1]
b_adjoint_new = b_adjoint + tmp_adjoint
a_adjoint_new = (b_adjoint_new, b_adjoint)

This works for
1. nested tuples: a = (b, (c, d))
2. another tuple var in tuple: a = (b, c), e = (a, d)

----> Checkpointing

@tvm.script.ir_module
class InputModule:
    @R.function
    def main_adjoint(x: R.Tensor((1, 1, 16, 16), "float32")):
        R.func_attr({"checkpoints": [(z1, z2), (z3, z4)]})
        with R.dataflow():
            z1 = f(x, x)
            z2 = f(z1, x)
            z3 = f(z2, x)
            z4 = f(z3, x)
            z5 = f(z4, x)
            z5_adjoint = R.ones_like(z5)

            z3_1 = f(z2, x)
            z4_1 = f(z3_1, x)

            z4_adjoint = f1(z5_adjoint, z4_1, x)
            x_adjoint_1 = f2(z5_adjoint, z4_1, x)

            z3_adjoint = f1(z4_adjoint, z3_1, x)
            x_adjoint_2_0 = f2(z4_adjoint, z3_1, x)
            x_adjoint_2 = x_adjoint_1 + x_adjoint_2_0

            z1_1 = f(x, x)
            z2_1 = f(z1_1, x)

            z2_adjoint = f1(z3_adjoint, z2_1, x)
            x_adjoint_3_0 = f2(z3_adjoint, z2_1, x)
            x_adjoint_3 = x_adjoint_2 + x_adjoint_3_0

            z1_adjoint = f1(z2_adjoint, z1_1, x)
            x_adjoint_4_0 = f2(z2_adjoint, z1_1, x)
            x_adjoint_4 = x_adjoint_3 + x_adjoint_4_0

            x_adjoint_5_0 = f1(z1_adjoint, x, x)
            x_adjoint_5 = x_adjoint_4 + x_adjoint_5_0
            x_adjoint_6_0 = f2(z1_adjoint, x, x)
            x_adjoint_6 = x_adjoint_5 + x_adjoint_6_0
            R.output(z5, x_adjoint_6)
        return z5, x_adjoint_6
