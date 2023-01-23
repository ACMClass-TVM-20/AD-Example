from __future__ import annotations
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm import topi, relax, te
from tvm.script.parser import ir as I, relax as R, tir as T
from tvm.relax.testing import dump_ast
from tvm.tir import PrimFunc

from tvm import te

# A = te.placeholder((10, 5), "float32", name="A")
# B = te.placeholder((10, 5), "float32", name="B")
# C = topi.add(A, B)
# te_func = te.create_prim_func([A, B, C]).with_attr({"global_symbol": "add"})
# te_func.show()
# print(te_func.script())
# module = tvm.IRModule.from_expr(te_func)
# # module.show()


# a_nd = tvm.nd.array(np.ones((10, 5)).astype(np.float32))
# b_nd = tvm.nd.array(np.ones((10, 5)).astype(np.float32))
# c_nd = tvm.nd.array(np.zeros((10, 5)).astype(np.float32))

# func = tvm.build(te_func, [A, B, C], target="llvm")

# func(a_nd, b_nd, c_nd)
# # print(c_nd.numpy())

# @I.ir_module
# class Mod1:
#     @T.prim_func
#     def add(A: T.Buffer[(10, 5), "float32"], B: T.Buffer[(10, 5), "float32"], T_add: T.Buffer[(10, 5), "float32"]):
#         # function attr dict
#         T.func_attr({"global_symbol": "add", "tir.noalias": True})
#         # body
#         # with T.block("root")
#         for i0, i1 in T.grid(10, 5):
#             with T.block("T_add"):
#                 ax0, ax1 = T.axis.remap("SS", [i0, i1])
#                 T.reads(A[ax0, ax1], B[ax0, ax1])
#                 T.writes(T_add[ax0, ax1])
#                 T_add[ax0, ax1] = A[ax0, ax1] + B[ax0, ax1]
# Mod1.show()
# print(Mod1.__str__())
# print(Mod1.__repr__())


def test(A):
    B = te.compute(shape=(3, 3), fcompute=lambda i, j: A[i+2*j], name="B")
    C = te.compute(shape=(), fcompute=lambda: B[0][0], name="C")
    return C

A = te.placeholder((7,), name="A")
B = test(A)
db = te.gradient(B, A)
te_func1 = te.create_prim_func([A, B]).with_attr({"global_symbol": "add"})
te_func2 = te.create_prim_func([A, db[0]]).with_attr({"global_symbol": "add"})
te_func1.show()
te_func2.show()
