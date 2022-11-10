from __future__ import annotations  # must import to defer parsing of annotations
import pytest
import tvm
from tvm import tir, relay, relax
from tvm.ir import assert_structural_equal

import tvm.script
from tvm.script import tir as T, relax as R

from tvm.relax.testing.ast_printer import dump_ast

# @tvm.script.ir_module
# class Before:
#     @R.function
#     def main(x1: Tensor((1, 10), "float32"),
#                 y1: Tensor((1, 10), "float32"),
#                 x2: Tensor((1, 10), "float32"),
#                 y2: Tensor((1, 10), "float32"),
#                 z: Tensor((1, 10), "float32")):
#         with R.dataflow():
#             t = ((x1, y1), (x2, y2))
#             t0 = t[0]
#             t1 = t[1]
#             t00 = t0[0]
#             t01 = t0[1]
#             t10 = t1[0]
#             t11 = t1[1]
#             lv1 = relax.add(t00, t01)
#             lv2 = relax.sub(t11, lv1)
#             lv3 = relax.multiply(lv2, t10)
#             loss = relax.nn.softmax_cross_entropy(lv3, z)
#             R.output(loss)
#         return loss

# @tvm.script.ir_module
# class Before:
#     @R.function
#     def main(y: Tensor((10, 5), "float32")):
#         with R.dataflow():
#             z0 = (y, y)
#             R.output(z0)
#         return z0

@tvm.script.ir_module
class Before:
    @R.function
    def main(y: Tensor((10, 5), "float32")):
        with R.dataflow():
            z0 = (y, y)
            R.output(z0)
        return z0

Before.show()
NormalizedTest = relax.transform.Normalize()(Before)
NormalizedTest.show()

ast = dump_ast(NormalizedTest["main"].body.blocks[0])
print(ast)