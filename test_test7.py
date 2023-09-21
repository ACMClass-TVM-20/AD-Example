import tvm
from tvm import relax, tir, te, topi
from tvm.ir.module import IRModule
from tvm.relax.analysis import estimate_memory_usage
from tvm.relax.block_builder import BlockBuilder
from tvm.relay import GlobalVar
from tvm.target.target import Target
import tvm.testing
from tvm.script.parser import relax as R, tir as T, ir as I
import pytest
import numpy as np

from tvm.relax.transform.tuning_api import Trace
from tvm.tir.schedule.schedule import Schedule

from tvm.relax.dpl.pattern import is_op, wildcard
import torch
import tvm.dlight as dl


# (shape1, shape3), (shape2, shape3)
shape1 = 2048
shape2 = 4096
orig_shape3 = 4096
tile = 128
uint_bits = 32
quant_bits = 4
one_uint_size = uint_bits // quant_bits
scale_shape3 = orig_shape3 // tile
quant_shape3 = orig_shape3 // one_uint_size
inner_shape3 = quant_shape3 // tile



def _tir_packed_uint_to_int(nbit, val, pos, dtype):
    max_int_value = (1 << (nbit - 1)) - 1
    return ((val >> (pos.astype("uint32") * tir.const(nbit, "uint32"))) & tir.const((1 << nbit) - 1, "uint32")).astype(dtype) - tir.const(max_int_value, dtype)

@I.ir_module
class Module:
    @T.prim_func
    def func(
        x: T.Buffer((shape1, quant_shape3), "uint32"),
        y: T.Buffer((shape2, quant_shape3), "uint32"),
        xs: T.Buffer((shape1, scale_shape3), "float16"),
        ys: T.Buffer((shape2, scale_shape3), "float16"),
        result: T.Buffer((shape1, shape2), "float16"),
    ):
        value_accu = T.alloc_buffer((shape1, shape2), "int32")
        for i, j in T.grid(shape1, shape2):
            result[i, j] = T.float16(0)
        for out in range(scale_shape3):
            for i, j in T.grid(shape1, shape2):
                for k in range(tile):
                    value_accu[i, j] = value_accu[i, j] + (
                        _tir_packed_uint_to_int(quant_bits, x[i, out * tile + k // one_uint_size], k % one_uint_size, "int32")
                        * _tir_packed_uint_to_int(quant_bits, y[j, out * tile + k // one_uint_size], k % one_uint_size, "int32")
                    )
                result[i, j] += T.Cast("float16", value_accu[i, j]) * xs[i, out] * ys[j, out]

    @R.function
    def main(x: R.Tensor((shape1, quant_shape3), "uint32"),
        y: R.Tensor((shape2, quant_shape3), "uint32"),
        xs: R.Tensor((shape1, scale_shape3), "float16"),
        ys: R.Tensor((shape2, scale_shape3), "float16")):
        cls = Module
        result = R.call_tir(cls.func, (x, y, xs, ys))


Module.show()



# mod1 = relax.transform.Normalize()(mod)

# mod1.show(None, False)
# print(mod1["main"].params[0].name_hint)
# print(mod1["main"].params[1].name_hint)

# mod = relax.transform.Gradient("main")(mod)
# mod.show(None, False)
# mod["main"] = mod["main"].without_attr("global_symbol")
# mod = relax.transform.DeadCodeElimination(["main_adjoint"])(mod)
# mod.show(None, False)
mod = relax.transform.LegalizeOps()(mod)
# mod = relax.get_pipeline()(mod)
# mod.show(None, False)
# assert relax.analysis.well_formed(mod)

# target, dev = "llvm", tvm.cpu()
target, dev = tvm.target.Target("apple/m1-gpu-restricted"), tvm.metal()
# target, dev = tvm.target.Target("nvidia/geforce-rtx-3080"), tvm.cuda()
# work_dir = "/home/yxdong/relax-mlcai/other-repos/AD-Example/tmp/tune"
# # with tempfile.TemporaryDirectory() as work_dir:
# with target, tvm.transform.PassContext(trace=Trace(mod)):
#     mod = tvm.transform.Sequential(
#         [
#             relax.transform.MetaScheduleTuneIRMod(
#                 params={}, work_dir=work_dir, max_trials_global=8
#             ),
#             relax.transform.MetaScheduleApplyDatabase(work_dir),
#         ]
#     )(mod)
# assert relax.analysis.well_formed(mod)
# mod.show(None, False)

with target, tvm.transform.PassContext(trace=Trace(mod)):
    # mod_deploy = dl.ApplyDefaultSchedule(dl.gpu.GeneralReduction())(mod)
    mod_deploy = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
    # mod_deploy = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
mod.show()
# mod_deploy.show(None, False)

# target, dev = "llvm", tvm.cpu()
# ex = relax.build(mod, target)
# vm = relax.VirtualMachine(ex, dev)
# vm["func"]
# inputs = [np.random.rand(5, 30000).astype(np.float16), np.random.rand(30000, 5).astype(np.float16)]

# res = vm["main"](*[tvm.nd.array(x, dev) for x in inputs])
# print(res.numpy())
# res_np = inputs[0] @ inputs[1]
# print(res_np, res_np.dtype)
# res_np_fp32 = inputs[0].astype("float32") @ inputs[1].astype("float32")
# print(res_np_fp32, res_np_fp32.dtype)
# print(res_np_fp32.astype("float16"))
# print(torch.tensor(inputs[0]).to("mps") @ torch.tensor(inputs[1]).to("mps"))


# use_decl_buffer = False
# def apply_decl_buffer(*args, **kwargs):
#     if use_decl_buffer:
#         return T.decl_buffer(*args, **kwargs)
#     else:
#         return T.Buffer(*args, **kwargs)

# @T.prim_func
# def before(
#     A: T.Buffer((128, 128), "float32"), C: T.Buffer((8, 32, 8, 8), "float32")
# ) -> None:
#     B = T.alloc_buffer((128, 128))
#     for i, j in T.grid(128, 128):
#         with T.block("B"):
#             vi, vj = T.axis.remap("SS", [i, j])
#             B[vi, vj] = A[vi, vj] * 2.0
#     for i, j, k, l in T.grid(8, 32, 8, 8):
#         with T.block("C"):
#             vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
#             C[vi, vj, vk, vl] = B[
#                 ((((vi * 32) + vj) * 8 + vk) * 8 + vl) // 128,
#                 ((((vi * 32) + vj) * 8 + vk) * 8 + vl) % 128,
#             ]


# @T.prim_func
# def expected(A: T.Buffer([4, 256], "float32"), C: T.Buffer([4, 256], "float32")):
#     offset_ptr = T.allocate_const([1.0, 2.0, 3.0, 4.0], dtype="float32", extents=[4])
#     offset = apply_decl_buffer([4], data=offset_ptr)
#     for i, j in T.grid(4, 256):
#         with T.block("compute_C"):
#             vi, vj = T.axis.remap("SS", [i, j])
#             C[vi, vj] = (10.0 * vi + offset[vi]) + 100.0 * vj

# sch = tir.Schedule(before, debug_mask="all")
# block = sch.get_block("B")
# sch.compute_inline(block)
# after = sch.mod["main"]
# after.show(None, False)
