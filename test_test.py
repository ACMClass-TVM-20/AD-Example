import numpy as np
import numpy.random
import pytest
import tvm
from tvm._ffi.base import TVMError
import tvm.script
import tvm.testing
from tvm import relax
from tvm import relax as rx
from tvm import te, tir
from tvm.ir.base import assert_structural_equal
from tvm.relax import Function, Var
from tvm.relax.block_builder import BlockBuilder
from tvm.relax.expr import Expr
from tvm.relax.op import add, divide, multiply, sqrt, subtract
from tvm.relax.struct_info import TensorStructInfo
from tvm.relax.transform import LegalizeOps
from tvm.runtime.container import tuple_object
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T
from tvm.tir.expr import IntImm
from tvm.topi.nn.utils import get_pad_tuple



@tvm.script.ir_module
class NormalModule:
    @R.function
    def main(x0: R.Tensor((3, 3), "float32"), x1: R.Tensor((3, 3), "float32")):
        with R.dataflow():
            gv = R.sum(x0)
            R.output(gv)
        return gv

    @T.prim_func
    def sum(
        rxplaceholder: T.Buffer[(T.int64(3), T.int64(3)), "float32"],
        rxplaceholder_red: T.Buffer[(), "float32"],
    ):
        T.func_attr({"tir.noalias": True})
        for k0, k1 in T.grid(T.int64(3), T.int64(3)):
            with T.block("rxplaceholder_red"):
                v_k0, v_k1 = T.axis.remap("RR", [k0, k1])
                T.reads(rxplaceholder[v_k0, v_k1])
                T.writes(rxplaceholder_red[()])
                with T.init():
                    rxplaceholder_red[()] = T.float32(0)
                rxplaceholder_red[()] = (rxplaceholder_red[()] + rxplaceholder[v_k0, v_k1])

# # NLLLoss.show()
# lowered_mod = LegalizeOps()(NormalModule)
# lowered_mod.show()
# ex = relax.vm.build(lowered_mod, target="llvm")
# vm = relax.VirtualMachine(ex, tvm.cpu())
# u_in = tvm.nd.array(numpy.ones((3, 2, 8, 8)).astype(np.float32))
# x_in = tvm.nd.array(numpy.random.rand(3, 2, 10, 10).astype(np.float32))

# print("grad:\n", u_in.numpy(), "\nx:\n", x_in.numpy(), sep='')

# res = vm["main"](u_in, x_in)
# # print(res.numpy())
# print("res:\n",res[0].numpy())
# print("grad res:\n", res[1].numpy())
