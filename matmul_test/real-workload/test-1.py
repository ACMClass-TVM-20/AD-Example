"""b=4
A100: 868.351 us
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                                 Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  --------------------------------------------------------------------------------------------------
     69.5          2426413          3  808804.3  808666.0    807833    809914       1047.4  fused_fused_decode3_fused_NT_matmul21_cast21_add5_silu2_kernel_1
     20.6           718778          1  718778.0  718778.0    718778    718778          0.0  sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize192x128x32_stage3_warpsize4x2x1_tensor16x8x16_kernel
      7.5           262560          3   87520.0   86272.0     84064     92224       4220.7  fused_fused_decode3_fused_NT_matmul21_cast21_add5_silu2_kernel
      2.4            84575          1   84575.0   84575.0     84575     84575          0.0  dequantize_kernel
b=1
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     56.4           724442          3  241480.7  240671.0    240669    243102       1404.1  fused_fused_decode3_fused_NT_matmul21_cast21_add5_silu2_kernel_1
     20.4           262525          3   87508.3   87103.0     84063     91359       3664.8  fused_fused_decode3_fused_NT_matmul21_cast21_add5_silu2_kernel
     16.5           212191          1  212191.0  212191.0    212191    212191          0.0  void cutlass::Kernel<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_tn_align8>(T1::Params)
      6.6            85279          1   85279.0   85279.0     85279     85279          0.0  dequantize_kernel
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
topi.matmul
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from dequantization.quantize import quantize_param, dequantize_param_optimize, q3f16_1, q4f16_1


def flush():
    cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")
    cache.zero_()


@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_fused_decode3_fused_NT_matmul21_cast21_add5_silu2(
        lv2608: T.Buffer((T.int64(11008), T.int64(512)), "uint32"),
        lv2609: T.Buffer((T.int64(11008), T.int64(128)), "float16"),
        p_lv7330: T.handle,
        p_output0: T.handle,
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        b = T.int64()
        lv7330 = T.match_buffer(p_lv7330, (b, T.int64(512), T.int64(4096)), "float16")
        p_output0_intermediate = T.match_buffer(
            p_output0, (b, T.int64(512), T.int64(11008)), "float16"
        )
        p_output0_intermediate_1 = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
        var_NT_matmul_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(11008)))
        for i, j in T.grid(T.int64(11008), T.int64(4096)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv2608[v_i, v_j // T.int64(8)], lv2609[v_i, v_j // T.int64(32)])
                T.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (
                    T.Cast(
                        "float16",
                        T.bitwise_and(
                            T.shift_right(
                                lv2608[v_i, v_j // T.int64(8)],
                                T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4),
                            ),
                            T.uint32(15),
                        ),
                    )
                    - T.float16(7)
                ) * lv2609[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(b, T.int64(512), T.int64(11008), T.int64(4096)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv7330[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[
                    v_i0, v_i1, v_i2
                ] + T.Cast("float32", lv7330[v_i0, v_i1, v_k]) * T.Cast(
                    "float32", p_output0_intermediate_1[v_i2, v_k]
                )
        for i0, i1, i2 in T.grid(b, T.int64(512), T.int64(11008)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
                p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast(
                    "float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                )

    @R.function
    def main(
        w1: R.Tensor((11008, 512), "uint32"),
        w2: R.Tensor((11008, 128), "float16"),
        x: R.Tensor(("b", 512, 4096), "float16"),
    ):
        cls = Module
        b = T.int64()
        out = R.call_tir(
            cls.fused_fused_decode3_fused_NT_matmul21_cast21_add5_silu2,
            (w1, w2, x),
            R.Tensor((b, 512, 11008), "float16"),
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

b = 4
s = 512

atol, rtol = 1e-3, 1e-3

inputs = [
    torch.randn(11008, 4096, dtype=torch.float16).cuda(),
    torch.randn(b, s, 4096, dtype=torch.float16).cuda(),
    # torch.zeros(b, s, 11008, dtype=torch.float16).cuda(),
]
tvm_inputs = [tvm.nd.array(x.detach().cpu().numpy(), dev) for x in inputs]
quantized_param_tvm = quantize_param(tvm_inputs[0].copyto(tvm.cpu()), q4f16_1)
tvm_quantized_inputs = [
    quantized_param_tvm[0].copyto(dev),
    quantized_param_tvm[1].copyto(dev),
] + tvm_inputs[1:]

tvm_res = vm["main"](*tvm_quantized_inputs)

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

dequantized_param_tvm = dequantize_param_optimize(tvm_quantized_inputs[:2], (11008, 4096), q4f16_1)
dequantized_param = torch.tensor(dequantized_param_tvm.numpy()).cuda()
torch_res_0 = inputs[1] @ dequantized_param.T


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
