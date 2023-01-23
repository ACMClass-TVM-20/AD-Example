import pytest
import tvm
from tvm import relax
from tvm.error import DiagnosticError
from tvm.relax.transform import OperatorLegalizer
from tvm.script.parser import ir as I, relax as R, tir as T
import tvm.testing

import itertools
import numpy as np

@I.ir_module
class Conv2d:
	@R.function
	def main(
		x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
	) -> R.Tensor(None, "float32", ndim=4):
		gv = R.conv2d(x, w, padding=1, kernel_size=[3, 3])
		return gv
Conv2d.show()
mod = OperatorLegalizer(Conv2d).transform()
mod.show()

# padding = 0
# @tvm.script.ir_module
# class Module:
#     @R.function
#     def main(x: Tensor((2, 3, 28, 28), "float32"), w: Tensor((4, 3, 3, 3), "float32")) -> Tensor(None, "float32", ndim = 4):
#         # block 0
#         gv = R.call_tir(conv2d, (x, w), (2, 4, 26, 26), dtype="float32")
#         return gv

#     @T.prim_func
#     def conv2d(rxplaceholder: T.Buffer[(2, 3, 28, 28), "float32"], rxplaceholder_1: T.Buffer[(4, 3, 3, 3), "float32"], conv2d_nchw: T.Buffer[(2, 4, 26, 26), "float32"]) -> None:
#         # function attr dict
#         T.func_attr({"global_symbol": "conv2d", "tir.noalias": True})
#         # body
#         # with T.block("root")
#         pad_temp = T.alloc_buffer([2, 3, 28, 28], dtype="float32")
#         for i0, i1, i2, i3 in T.grid(2, 3, 28, 28):
#             with T.block("pad_temp"):
#                 i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
#                 T.reads(rxplaceholder[i0_1, i1_1, i2_1, i3_1])
#                 T.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
#                 pad_temp[i0_1, i1_1, i2_1, i3_1] = rxplaceholder[i0_1, i1_1, i2_1, i3_1]
#         for i0, i1, i2, i3, i4, i5, i6 in T.grid(2, 4, 26, 26, 3, 3, 3):
#             with T.block("conv2d_nchw"):
#                 nn, ff, yy, xx, rc, ry, rx = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
#                 T.reads(pad_temp[nn, rc, yy + ry, xx + rx], rxplaceholder_1[ff, rc, ry, rx])
#                 T.writes(conv2d_nchw[nn, ff, yy, xx])
#                 with T.init():
#                     conv2d_nchw[nn, ff, yy, xx] = T.float32(0)
#                 conv2d_nchw[nn, ff, yy, xx] = conv2d_nchw[nn, ff, yy, xx] + pad_temp[nn, rc, yy + ry, xx + rx] * rxplaceholder_1[ff, rc, ry, rx]


# padding = 1
# @tvm.script.ir_module
# class Module:
#     @R.function
#     def main(x: Tensor((2, 3, 28, 28), "float32"), w: Tensor((4, 3, 3, 3), "float32")) -> Tensor(None, "float32", ndim = 4):
#         # block 0
#         gv = R.call_tir(conv2d, (x, w), (2, 4, 28, 28), dtype="float32")
#         return gv

#     @T.prim_func
#     def conv2d(rxplaceholder: T.Buffer[(2, 3, 28, 28), "float32"], rxplaceholder_1: T.Buffer[(4, 3, 3, 3), "float32"], conv2d_nchw: T.Buffer[(2, 4, 28, 28), "float32"]) -> None:
#         # function attr dict
#         T.func_attr({"global_symbol": "conv2d", "tir.noalias": True})
#         # body
#         # with T.block("root")
#         pad_temp = T.alloc_buffer([2, 3, 30, 30], dtype="float32")
#         for i0, i1, i2, i3 in T.grid(2, 3, 30, 30):
#             with T.block("pad_temp"):
#                 i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
#                 T.reads(rxplaceholder[i0_1, i1_1, i2_1 - 1, i3_1 - 1])
#                 T.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
#                 pad_temp[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i2_1 and i2_1 < 29 and 1 <= i3_1 and i3_1 < 29, rxplaceholder[i0_1, i1_1, i2_1 - 1, i3_1 - 1], T.float32(0), dtype="float32")
#         for i0, i1, i2, i3, i4, i5, i6 in T.grid(2, 4, 28, 28, 3, 3, 3):
#             with T.block("conv2d_nchw"):
#                 nn, ff, yy, xx, rc, ry, rx = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
#                 T.reads(pad_temp[nn, rc, yy + ry, xx + rx], rxplaceholder_1[ff, rc, ry, rx])
#                 T.writes(conv2d_nchw[nn, ff, yy, xx])
#                 with T.init():
#                     conv2d_nchw[nn, ff, yy, xx] = T.float32(0)
#                 conv2d_nchw[nn, ff, yy, xx] = conv2d_nchw[nn, ff, yy, xx] + pad_temp[nn, rc, yy + ry, xx + rx] * rxplaceholder_1[ff, rc, ry, rx]
