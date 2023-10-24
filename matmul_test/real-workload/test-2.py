"""n = 512
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                                  Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     43.9           364414          1  364414.0  364414.0    364414    364414          0.0  fused_fused_decode4_fused_NT_matmul6_cast_add_add_kernel_1
     32.0           265630          1  265630.0  265630.0    265630    265630          0.0  ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_tn
     11.0            91135          1   91135.0   91135.0     91135     91135          0.0  fused_fused_decode4_fused_NT_matmul6_cast_add_add_kernel
     10.1            84095          1   84095.0   84095.0     84095     84095          0.0  dequantize_kernel
      2.9            24096          2   12048.0   12048.0      9248     14848       3959.8  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::Half>, at::…
n=1024
Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                                  Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     47.3           569532          1  569532.0  569532.0    569532    569532          0.0  fused_fused_decode4_fused_NT_matmul6_cast_add_add_kernel_1
     34.9           419549          1  419549.0  419549.0    419549    419549          0.0  sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x160x32_stage3_warpsize2x4x1_tensor16x8x16_kernel
      7.6            91968          1   91968.0   91968.0     91968     91968          0.0  fused_fused_decode4_fused_NT_matmul6_cast_add_add_kernel
      7.0            84575          1   84575.0   84575.0     84575     84575          0.0  dequantize_kernel
      3.1            37664          2   18832.0   18832.0     15552     22112       4638.6  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::Half>, at::…
n=2048
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                                  Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     49.1           903033          1  903033.0  903033.0    903033    903033          0.0  fused_fused_decode4_fused_NT_matmul6_cast_add_add_kernel_1
     36.9           678171          1  678171.0  678171.0    678171    678171          0.0  ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_tn
      5.0            92288          1   92288.0   92288.0     92288     92288          0.0  fused_fused_decode4_fused_NT_matmul6_cast_add_add_kernel
      4.7            86528          1   86528.0   86528.0     86528     86528          0.0  dequantize_kernel
      4.3            79871          2   39935.5   39935.5     37248     42623       3800.7  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::Half>, at::…
"""
import os
import sys
from typing import List

from lift_tir_global_buffer_alloc import LiftTIRGlobalBufferAlloc

import tvm
from tvm import relax, tir, te, topi
from tvm.dlight.gpu import fallback
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

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from dequantization.quantize import quantize_param, dequantize_param_optimize, q3f16_1, q4f16_1


@I.ir_module
class Module:
    @T.prim_func
    def fused_fused_decode4_fused_NT_matmul6_cast_add_add(
        lv37: T.Buffer((T.int64(4096), T.int64(1376)), "uint32"),
        lv38: T.Buffer((T.int64(4096), T.int64(344)), "float16"),
        p_lv105: T.handle,
        p_lv41: T.handle,
        p_lv26: T.handle,
        p_output0: T.handle,
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv105 = T.match_buffer(p_lv105, (T.int64(1), n, T.int64(11008)), "float16")
        lv41 = T.match_buffer(p_lv41, (T.int64(1), n, T.int64(4096)), "float16")
        lv26 = T.match_buffer(p_lv26, (T.int64(1), n, T.int64(4096)), "float16")
        p_output0_intermediate_real = T.match_buffer(
            p_output0, (T.int64(1), n, T.int64(4096)), "float32"
        )
        p_output0_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
        p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)))
        var_compute_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
        for i, j in T.grid(T.int64(4096), T.int64(11008)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv37[v_i, v_j // T.int64(8)], lv38[v_i, v_j // T.int64(32)])
                T.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (
                    T.Cast(
                        "float16",
                        T.bitwise_and(
                            T.shift_right(
                                lv37[v_i, v_j // T.int64(8)],
                                T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4),
                            ),
                            T.uint32(15),
                        ),
                    )
                    - T.float16(7)
                ) * lv38[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(11008)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv105[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[
                    v_i0, v_i1, v_i2
                ] + T.Cast("float32", lv105[v_i0, v_i1, v_k]) * T.Cast(
                    "float32", p_output0_intermediate_1[v_i2, v_k]
                )
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
                var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast(
                    "float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                )
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_compute_intermediate[v_ax0, v_ax1, v_ax2], lv41[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = (
                    var_compute_intermediate[v_ax0, v_ax1, v_ax2] + lv41[v_ax0, v_ax1, v_ax2]
                )
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv26[v_ax0, v_ax1, v_ax2], var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                    lv26[v_ax0, v_ax1, v_ax2] + var_T_add_intermediate[v_ax0, v_ax1, v_ax2]
                )
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("T_cast"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                p_output0_intermediate_real[v_ax0, v_ax1, v_ax2] = T.Cast(
                    "float32", p_output0_intermediate[v_ax0, v_ax1, v_ax2]
                )

    @R.function
    def main(
        w1: R.Tensor((4096, 1376), "uint32"),
        w2: R.Tensor((4096, 344), "float16"),
        x: R.Tensor((1, "n", 11008), "float16"),
        y: R.Tensor((1, "n", 4096), "float16"),
        z: R.Tensor((1, "n", 4096), "float16"),
    ):
        cls = Module
        n = T.int64()
        out = R.call_tir(
            cls.fused_fused_decode4_fused_NT_matmul6_cast_add_add,
            (w1, w2, x, y, z),
            R.Tensor((1, n, 4096), "float32"),
        )
        return out


target, dev = tvm.target.Target("nvidia/nvidia-a100"), tvm.cuda()

cur_path = os.path.dirname(os.path.abspath(__file__))
before_path = os.path.join(cur_path, "before.py")
after_path = os.path.join(cur_path, "after.py")
suffix_map = {
    "cuda": ".cu",
    "metal": ".mtl",
    "cpu": ".ll",
}
dump_path = os.path.join(cur_path, "build" + suffix_map[target.kind.default_keys[0]])
ex_path = os.path.join(cur_path, "build" + suffix_map[target.kind.default_keys[0]] + ".so")
cubin_path = os.path.join(cur_path, "build.cubin")


mod = Module
if target.kind.name == "cuda":
    with target, tvm.transform.PassContext(trace=Trace(mod)):
        mod = dl.ApplyDefaultSchedule(dl.gpu.matmul.MatmulTensorizationMMA())(mod)
        # mod = dl.ApplyDefaultSchedule(dl.gpu.matmul.Matmul())(mod)
mod = LiftTIRGlobalBufferAlloc()(mod)
print(mod.script(), file=open(after_path, "w"))
print("<schedule done>")

# build
# func = next(mod.functions.values())
with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
    ex = relax.build(mod, target=target)
if target.kind.name == "cuda":
    print(ex.mod.imported_modules[0].imported_modules[0].get_source(), file=open(dump_path, "w"))
ex.export_library(ex_path)
vm = relax.VirtualMachine(ex, dev, profile=True)

n = 2048

atol, rtol = 1e-3, 1e-3

weight_torch = torch.randn(4096, 11008, dtype=torch.float16).cuda()
weight_tvm_cpu = tvm.nd.array(weight_torch.detach().cpu().numpy(), tvm.cpu())
inputs_torch = [
    torch.randn(1, n, 11008, dtype=torch.float16).cuda(),
    torch.randn(1, n, 4096, dtype=torch.float16).cuda(),
    torch.randn(1, n, 4096, dtype=torch.float16).cuda(),
]
inputs_tvm = [tvm.nd.array(x.detach().cpu().numpy(), dev) for x in inputs_torch]

weight_quantized_tvm_cpu = quantize_param(weight_tvm_cpu, q4f16_1)
weight_quantized_tvm = [i.copyto(dev) for i in weight_quantized_tvm_cpu]

tvm_res = vm["main"](*weight_quantized_tvm, *inputs_tvm)

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
weight_dequantized_tvm = dequantize_param_optimize(weight_quantized_tvm, (4096, 11008), q4f16_1)
weight_dequantized_torch = torch.tensor(weight_dequantized_tvm.numpy()).cuda()
torch_res_0 = inputs_torch[0] @ weight_dequantized_torch.T + inputs_torch[1] + inputs_torch[2]

close = np.allclose(torch_res_0.detach().cpu().numpy(), tvm_res.numpy(), atol=atol, rtol=rtol)
if not close:
    print("torch:\n", torch_res_0.detach().cpu().numpy())
    print("tvm:\n", tvm_res.numpy())
    assert close

print("<correctness check done>")

# report = vm.profile("main", *tvm_quantized_inputs)
# print(report)

# operator_call, operator_tm = None, None
# for op in report.calls:
#     if operator_call is None or op["Duration (us)"].microseconds > operator_tm:
#         operator_call, operator_tm = op, op["Duration (us)"].microseconds
# print(operator_call)


# tflops = b * s * 11008 * 4096 * 2 / operator_tm / 1e6
# print(f"Op latency: {operator_tm} us, TFlops: {tflops}")
# print("<performance check done>")
