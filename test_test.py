import numpy as np
import pytest
import tvm
from tvm.ir.base import assert_structural_equal
from tvm.relax.block_builder import BlockBuilder
from tvm.relax.expr import Expr
from tvm.runtime.container import tuple_object
import tvm.script
from tvm import relax
from tvm.script.parser import ir as I, relax as R, tir as T
from tvm.relax.transform import LegalizeOps

import pytest
import tvm
import tvm.testing
from tvm import relax
from tvm.ir.base import assert_structural_equal
from tvm.script.parser import relax as R

import tvm
from tvm import relax as rx
from tvm.relax.struct_info import TensorStructInfo
from tvm.runtime.container import tuple_object
from tvm.relax.op import add, subtract, multiply, divide, sqrt
from tvm.relax import Var, Function
import numpy.random


@tvm.script.ir_module
class NLLLoss:
    @R.function
    def main(
        grad: R.Tensor((), "float32"),
        predictions: R.Tensor((3,), "float32"),
        targets: R.Tensor((), "int64"),
        weights: R.Tensor((3,), "float32"),
    ):
        gv1 = R.nn.nll_loss(predictions, targets, weights, reduction="mean")
        gv = R.nll_loss_backward_pred(
            grad, predictions, targets, weights, reduction="mean", ignore_index=-1
        )
        return (gv, gv1)

NLLLoss.show()
lowered_mod = LegalizeOps()(NLLLoss)
lowered_mod.show()
ex = relax.vm.build(lowered_mod, target="llvm")
print(ex.as_text())
vm = relax.VirtualMachine(ex, tvm.cpu())
u_in = tvm.nd.array(numpy.ones(()).astype(np.float32))
x_in = tvm.nd.array(numpy.random.rand(3).astype(np.float32))
y_in = tvm.nd.array(numpy.random.randint(0, 3, size=()).astype(np.int64))
z_in = tvm.nd.array(numpy.random.randint(1, 5, size=(3,)).astype(np.float32))

print(u_in.numpy(), x_in.numpy(), y_in.numpy(), z_in.numpy())

res = vm["main"](u_in, x_in, y_in, z_in)
print(res[0].numpy())
print(res[1].numpy())
