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
from tvm.relax.expr_functor import mutator, PyExprMutator
from tvm.relax.transform.tuning_api import Trace
from tvm.tir.schedule.schedule import Schedule

from tvm.relax.dpl.pattern import is_op, wildcard
import torch
import tvm.dlight as dl

# bb = relax.BlockBuilder()
# dtype = "float32"
# x = relax.Var("x", R.Tensor((1, 2, 5, 5), dtype))
# w = relax.Var("x", R.Tensor((2, 2, 1, 1), dtype))


# with bb.function("main", [x, w]):
#     with bb.dataflow():
#         lv1 = bb.emit(relax.op.nn.conv2d(x, w))
#         lv2 = bb.emit(lv1 + R.const(1, dtype))
#         lv3 = bb.emit(lv1 + R.const(1, dtype))
#         gv = bb.emit_output((lv2, lv3))
#     bb.emit_func_output(gv)

# mod = bb.get()


# @I.ir_module
# class mod:
#     @R.function
#     def main(input: R.Tensor((3, 3), "float32"), input1: R.Tensor((3, 3), "float32")):
#         with R.dataflow():
#             # gv = R.reshape(input, (9,))
#             gv = R.matmul(input, input1)
#             R.output(gv)
#         return gv

    # @T.prim_func
    # def main(input: T.Buffer((), "int32"), output: T.Buffer((), "int32")):
    #     with T.block("T_add"):
    #         vi = T.axis.spatial(1, T.int64(0))
    #         T.reads(input[()])
    #         T.writes(output[()])
    #         output[()] = input[()] + 1


# mod1 = relax.transform.Normalize()(mod)

# mod1.show(None, False)
# print(mod1["main"].params[0].name_hint)
# print(mod1["main"].params[1].name_hint)

# mod = relax.transform.Gradient("main")(mod)
# mod.show(None, False)
# mod["main"] = mod["main"].without_attr("global_symbol")
# mod = relax.transform.DeadCodeElimination(["main_adjoint"])(mod)
# mod.show(None, False)
# mod.show()


# @tvm.transform.module_pass(opt_level=0, name="TestPass")
# class TestPass:
#     def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
#         @mutator
#         class TestMutator(PyExprMutator):
#             def __init__(self, mod: IRModule):
#                 super().__init__(mod)
#                 self.mod = mod

#             def transform(self) -> IRModule:
#                 for global_var, func in self.mod.functions.items():
#                     if not isinstance(func, relax.Function):
#                         continue
#                     updated_func = self.visit_expr(func)
#                     self.builder_.update_func(global_var, updated_func)
#                 return self.builder_.get()

#             # def visit_call_(self, call):
#             #     call = self.visit_expr_post_order(call)

#             #     if call.op != tvm.ir.Op.get("relax.call_tir_with_grad"):
#             #         return call

#             #     return relax.Call(tvm.ir.Op.get("relax.call_tir"), call.args, None, call.sinfo_args)

#             def visit_var_(self, var):
#                 return self.builder_.emit(var)

#         return TestMutator(mod).transform()

# mod = TestPass()(mod)
# mod = relax.transform.LegalizeOps()(mod)
# mod = relax.get_pipeline()(mod)
# mod.show(None, False)
# assert relax.analysis.well_formed(mod)

# target, dev = "llvm", tvm.cpu()
# target, dev = tvm.target.Target("apple/m1-gpu-restricted"), tvm.metal()
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

# with target, tvm.transform.PassContext(trace=Trace(mod)):
    #     # mod_deploy = dl.ApplyDefaultSchedule(dl.gpu.GeneralReduction())(mod)
    # mod_deploy = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
#     # mod_deploy = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
# mod.show()
# mod_deploy.show(None, False)

# target, dev = "llvm", tvm.cpu()
# ex = relax.build(mod, target)
# vm = relax.VirtualMachine(ex, dev)
# vm["func"]
# inputs = [np.random.rand(3, 3).astype(np.float32), np.random.rand(3, 3).astype(np.float32)]

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


@tvm.script.ir_module
class TransformedGlobalToSharedWithLocalStage:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (1024, 1024))
        B = T.match_buffer(b, (1024, 1024))
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block(""):
                            T.reads(A[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                            T.writes(B[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                            A_shared_dyn = T.alloc_buffer(
                                (128, 128), strides=(128, 1), scope="shared.dyn"
                            )
                            with T.block("A_shared"):
                                T.reads(A[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                                T.writes(A_shared_dyn[0:128, 0:128])
                                T.block_attr(
                                    {"auto_copy": 1, "local_stage": True, "vector_bytes": 16}
                                )
                                A_shared_dyn_local = T.alloc_buffer((16, 4), scope="local")
                                for ax0_ax1_fused_1 in T.thread_binding(8, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(
                                        32, thread="threadIdx.x"
                                    ):
                                        for ax0_ax1_fused_0_cache in range(16):
                                            for ax0_ax1_fused_3_cache in T.vectorized(4):
                                                A_shared_dyn_local[
                                                    ax0_ax1_fused_0_cache
                                                    * 8
                                                    * 32
                                                    * 4
                                                    // 128
                                                    % 128
                                                    // 8,
                                                    ax0_ax1_fused_3_cache % 128,
                                                ] = A[
                                                    bx * 128
                                                    + (
                                                        (
                                                            (
                                                                ax0_ax1_fused_0_cache * 8
                                                                + ax0_ax1_fused_1
                                                            )
                                                            * 32
                                                            + ax0_ax1_fused_2
                                                        )
                                                        * 4
                                                        + ax0_ax1_fused_3_cache
                                                    )
                                                    // 128
                                                    % 128,
                                                    by * 128
                                                    + (
                                                        (
                                                            (
                                                                ax0_ax1_fused_0_cache * 8
                                                                + ax0_ax1_fused_1
                                                            )
                                                            * 32
                                                            + ax0_ax1_fused_2
                                                        )
                                                        * 4
                                                        + ax0_ax1_fused_3_cache
                                                    )
                                                    % 128,
                                                ]
                                        for ax0_ax1_fused_0 in range(16):
                                            for ax0_ax1_fused_3 in T.vectorized(4):
                                                A_shared_dyn[
                                                    (
                                                        (
                                                            (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1)
                                                            * 32
                                                            + ax0_ax1_fused_2
                                                        )
                                                        * 4
                                                        + ax0_ax1_fused_3
                                                    )
                                                    // 128
                                                    % 128,
                                                    (
                                                        (
                                                            (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1)
                                                            * 32
                                                            + ax0_ax1_fused_2
                                                        )
                                                        * 4
                                                        + ax0_ax1_fused_3
                                                    )
                                                    % 128,
                                                ] = A_shared_dyn_local[
                                                    ax0_ax1_fused_0 * 8 * 32 * 4 // 128 % 128 // 8,
                                                    ax0_ax1_fused_3 % 128,
                                                ]
                            with T.block("B"):
                                T.reads(A_shared_dyn[0:128, 0:128])
                                T.writes(B[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                                for ax0 in range(128):
                                    for ax1 in range(128):
                                        B[bx * 128 + ax0, by * 128 + ax1] = A_shared_dyn[ax0, ax1]
TransformedGlobalToSharedWithLocalStage.show(None, False)
