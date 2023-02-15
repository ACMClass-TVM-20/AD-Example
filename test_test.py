import numpy as np
import numpy.random
import pytest
import tvm
from tvm._ffi.base import TVMError
from tvm.arith.analyzer import Analyzer
from tvm.relax.transform.transform import ToMixedPrecision
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
from tvm.tir.expr import IntImm
from tvm.topi.nn.utils import get_pad_tuple

from tvm.relay.op import tile


@I.ir_module
class Repeat:
    @R.function
    def main(x: R.Tensor((3, 2, 4), "float32")):
        gv = R.reshape(x, (-1, 0, 2)) # 6, 2, 2
        return gv
# NLLLoss.show()
lowered_mod = LegalizeOps()(Repeat)
lowered_mod.show()
ex = relax.vm.build(lowered_mod, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())
u_in = tvm.nd.array(np.array(range(3 * 2 * 4)).reshape(3, 2, 4).astype(np.float32))
# x_in = tvm.nd.array(numpy.random.rand(3, 4, 3, 3).astype(np.float32))
print(u_in.numpy())
res = vm["main"](u_in)
print(res.numpy())
