import numpy as np
import numpy.random
import pytest
import tvm
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

# @tvm.script.ir_module
# class NLLLoss:
#     @R.function
#     def main(
#         grad: R.Tensor((3, 2, 8, 8), dtype="float32"),
#         x: R.Tensor((3, 2, 10, 10), dtype="float32"),
#     ):
#         gv = R.nn.max_pool2d(x, (3, 3))
#         gv1 = R.max_pool2d_backward(grad, x, (3, 3))
#         return gv, gv1

# # NLLLoss.show()
# lowered_mod = LegalizeOps()(NLLLoss)
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

print(get_pad_tuple(3, (tir.Var("a", "int64"), tir.Var("b", "int64"))))

import torch.nn.modules.linear
