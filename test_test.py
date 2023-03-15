import numpy as np
import numpy.random
import pytest
import tvm
from tvm._ffi.base import TVMError
from tvm.arith.analyzer import Analyzer
import tvm.script
import tvm.testing
from tvm import relax
from tvm import relax as rx
from tvm import te, tir
from tvm.ir.base import assert_structural_equal
from tvm.relax import Function, Var
from tvm.relax.block_builder import BlockBuilder
from tvm.relax.expr import Expr
from tvm import topi, relay
from tvm.relax.op import add, divide, multiply, sqrt, subtract
from tvm.relax.struct_info import TensorStructInfo
from tvm.relax.transform import LegalizeOps
from tvm.runtime.container import tuple_object
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T
from tvm.tir.function import PrimFunc


c1 = R.const(np.zeros(3).astype(np.float32))
c2 = R.const(np.zeros(3).astype(np.float32))
c3 = R.const(np.zeros(3).astype(np.float32))

@tvm.script.ir_module
class Before:
    @R.function
    def main(x: R.Tensor((3,), "float32")):
        # block 0
        with R.dataflow():
            gv = x
            R.output(gv)
        return (gv, gv)
print(Before.attrs)
# assert_structural_equal(Expected, After)
# old_f = Module["main"]
# new_f = relax.utils.copy_with_new_vars(old_f)
# print(new_f)
lowered_mod = LegalizeOps()(Before)
lowered_mod.show(None, False)
ex = relax.build(lowered_mod, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())
u_in = tvm.nd.array(numpy.zeros((3,)).astype(np.float32))
# x_in = tvm.nd.array(numpy.random.rand(3, 4, 3, 3).astype(np.float32))
print(u_in.numpy())
res = vm["main"](u_in)
# print(res.numpy())
