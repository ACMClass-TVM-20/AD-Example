"""Test: fp16 mixed precision matmul mma
- Without unroll: 99.60745695626359
    - 208 regs
- With unroll: 144.07358026271964
- With pipeline: 96.57410868904708
    - 167 regs
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
    def fused_fused_decode4_transpose5_fused_NT_matmul21_cast21(
        lv4075: T.Buffer((T.int64(4096), T.int64(1376)), "uint32"),
        lv4076: T.Buffer((T.int64(4096), T.int64(344)), "float16"),
        p_lv10517_adjoint: T.handle,
        p_output0: T.handle,
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        b = T.int64()
        lv10517_adjoint = T.match_buffer(
            p_lv10517_adjoint, (b, T.int64(512), T.int64(4096)), "float16"
        )
        p_output0_intermediate = T.match_buffer(
            p_output0, (b, T.int64(512), T.int64(11008)), "float16"
        )
        # with T.block("root"):
        var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
        p_output0_intermediate_1 = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
        var_NT_matmul_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(11008)))
        for i, j in T.grid(T.int64(4096), T.int64(11008)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv4075[v_i, v_j // T.int64(8)], lv4076[v_i, v_j // T.int64(32)])
                T.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = (
                    T.Cast(
                        "float16",
                        T.bitwise_and(
                            T.shift_right(
                                lv4075[v_i, v_j // T.int64(8)],
                                T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4),
                            ),
                            T.uint32(15),
                        ),
                    )
                    - T.float16(7)
                ) * lv4076[v_i, v_j // T.int64(32)]
        for ax0, ax1 in T.grid(T.int64(11008), T.int64(4096)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(var_decode_intermediate[v_ax1, v_ax0])
                T.writes(p_output0_intermediate_1[v_ax0, v_ax1])
                p_output0_intermediate_1[v_ax0, v_ax1] = var_decode_intermediate[v_ax1, v_ax0]
        for i0, i1, i2, k in T.grid(b, T.int64(512), T.int64(11008), T.int64(4096)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv10517_adjoint[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[
                    v_i0, v_i1, v_i2
                ] + T.Cast("float32", lv10517_adjoint[v_i0, v_i1, v_k]) * T.Cast(
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
        w1: R.Tensor((4096, 1376), "uint32"),
        w2: R.Tensor((4096, 344), "float16"),
        x: R.Tensor(("b", 512, 4096), "float16"),
    ):
        cls = Module
        b = T.int64()
        out = R.call_tir(
            cls.fused_fused_decode4_transpose5_fused_NT_matmul21_cast21,
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
print("<build done>")

b = 4
s = 512

atol, rtol = 1e-3, 1e-3

weight_torch = torch.randn(4096, 11008, dtype=torch.float16).cuda()
weight_tvm_cpu = tvm.nd.array(weight_torch.detach().cpu().numpy(), tvm.cpu())
inputs_torch = [
    torch.randn(b, s, 4096, dtype=torch.float16).cuda(),
]
inputs_tvm = [tvm.nd.array(x.detach().cpu().numpy(), dev) for x in inputs_torch]

weight_quantized_tvm_cpu = quantize_param(weight_tvm_cpu, q4f16_1)
weight_quantized_tvm = [i.copyto(dev) for i in weight_quantized_tvm_cpu]

tvm_res = vm["main"](*weight_quantized_tvm, *inputs_tvm)

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
weight_dequantized_tvm = dequantize_param_optimize(weight_quantized_tvm, (4096, 11008), q4f16_1)
weight_dequantized_torch = torch.tensor(weight_dequantized_tvm.numpy()).cuda()
torch_res_0 = inputs_torch[0] @ weight_dequantized_torch

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
