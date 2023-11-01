# fp32 output


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

# fmt: off
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_matmul1_cast13(p_A: T.handle, p_B: T.handle, p_O: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        b = T.int64()
        A = T.match_buffer(p_A, (b, T.int64(4096), T.int64(4096)), "float16")
        B = T.match_buffer(p_B, (T.int64(4096), T.int64(4096)), "float16")
        O = T.match_buffer(p_O, (b, T.int64(4096), T.int64(4096)), "float32")
        # with T.block("root"):
        var_matmul_intermediate = T.alloc_buffer((b, T.int64(4096), T.int64(4096)))
        for i0, i1, i2, k in T.grid(b, T.int64(4096), T.int64(4096), T.int64(4096)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", A[v_i0, v_k, v_i1]) * T.Cast("float32", B[v_k, v_i2])
        for i0, i1, i2 in T.grid(b, T.int64(4096), T.int64(4096)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                O[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2]

    @R.function
    def main(
        x: R.Tensor(("b", 4096, 4096), "float16"),
        y: R.Tensor((4096, 4096), "float16"),
    ):
        cls = Module
        b = T.int64()
        out = R.call_tir(
            cls.fused_matmul1_cast13,
            (x, y),
            R.Tensor((b, 4096, 4096), "float32"),
        )
        return out
# fmt: on

# target, dev = tvm.target.Target("nvidia/geforce-rtx-4090"), tvm.cuda()
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

func = next(mod.functions.values())
with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
    ex = relax.build(mod, target=target)
if target.kind.name == "cuda":
    print(ex.mod.imported_modules[0].imported_modules[0].get_source(), file=open(dump_path, "w"))
ex.export_library(ex_path)
vm = relax.VirtualMachine(ex, dev, profile=True)
print("<build done>")

b = 4

atol, rtol = 1e-3, 1e-3

inputs_torch = [
    torch.randn(b, 4096, 4096, dtype=torch.float16).cuda(),
    torch.randn(4096, 4096, dtype=torch.float16).cuda(),
]
inputs_tvm = [tvm.nd.array(x.detach().cpu().numpy(), dev) for x in inputs_torch]

tvm_res = vm["main"](*inputs_tvm)

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch_res = inputs_torch[0].mT @ inputs_torch[1]

close = np.allclose(torch_res.detach().cpu().numpy(), tvm_res.numpy(), atol=atol, rtol=rtol)
if not close:
    print("torch:\n", torch_res.detach().cpu().numpy())
    print("tvm:\n", tvm_res.numpy())
    assert close

print("<correctness check done>")

report = vm.profile("main", *inputs_tvm)
print(report)

operator_call, operator_tm = None, None
for op in report.calls:
    if operator_call is None or op["Duration (us)"].microseconds > operator_tm:
        operator_call, operator_tm = op, op["Duration (us)"].microseconds
print(operator_call)


tflops = b * 4096 * 4096 * 4096 * 2 / operator_tm / 1e6
print(f"Op latency: {operator_tm} us, TFlops: {tflops}")
print("<performance check done>")
