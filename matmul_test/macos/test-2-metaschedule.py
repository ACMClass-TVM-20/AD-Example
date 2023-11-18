import os
import sys
from typing import List
from time import monotonic

import tvm
from tvm import relax, tir, te, topi
from tvm import meta_schedule as ms
from tvm.dlight.gpu import fallback
from tvm.ir.module import IRModule
from tvm.meta_schedule.database.json_database import JSONDatabase
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
from real_workload.lift_tir_global_buffer_alloc import LiftTIRGlobalBufferAlloc
from dequantization.quantize import quantize_param, dequantize_param_optimize, q3f16_1, q4f16_1


def flush_cache():
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="mps")
    cache.zero_()


# fmt: off
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def compute(B: T.Buffer((T.int64(4096), T.int64(4096)), "float16"), p_A: T.handle, p_O: T.handle):
        A = T.match_buffer(p_A, (T.int64(2048), T.int64(4096)), "float16")
        O = T.match_buffer(p_O, (T.int64(2048), T.int64(4096)), "float16")
        # with T.block("root"):
        O_intermediate = T.alloc_buffer((T.int64(2048), T.int64(4096)))
        for i0, i1, k in T.grid(T.int64(2048), T.int64(4096), T.int64(4096)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(A[v_i0, v_k], B[v_k, v_i1])
                T.writes(O_intermediate[v_i0, v_i1])
                with T.init():
                    O_intermediate[v_i0, v_i1] = T.float32(0)
                O_intermediate[v_i0, v_i1] = O_intermediate[v_i0, v_i1] + T.Cast("float32", A[v_i0, v_k]) * T.Cast("float32", B[v_k, v_i1])
        for i0, i1 in T.grid(T.int64(2048), T.int64(4096)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(O_intermediate[v_i0, v_i1])
                T.writes(O[v_i0, v_i1])
                O[v_i0, v_i1] = T.Cast("float16", O_intermediate[v_i0, v_i1])


    @R.function
    def main(
        w: R.Tensor((4096, 4096), "float16"),
        x: R.Tensor((2048, 4096), "float16"),
    ):
        cls = Module
        out = R.call_tir(
            cls.compute,
            (w, x),
            R.Tensor((2048, 4096), "float16"),
        )
        return out
# fmt: on

# target, dev = tvm.target.Target("nvidia/nvidia-a100"), tvm.cuda()
target, dev = tvm.target.Target("apple/m2-gpu"), tvm.metal()

cur_path = os.path.dirname(os.path.abspath(__file__))
before_path = os.path.join(cur_path, "before.py")
after_path = os.path.join(cur_path, "after.py")
trace_path = os.path.join(cur_path, "trace.py")
suffix_map = {
    "cuda": ".cu",
    "metal": ".mtl",
    "cpu": ".ll",
}
dump_path = os.path.join(cur_path, "build" + suffix_map[target.kind.default_keys[0]])
ex_path = os.path.join(cur_path, "build" + suffix_map[target.kind.default_keys[0]] + ".so")
cubin_path = os.path.join(cur_path, "build.cubin")


mod = Module
if target.kind.name != "llvm":
    work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tune")
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    # with target, tvm.transform.PassContext(trace=Trace(mod)):
    #     mod = tvm.transform.Sequential(
    #         [
    #             relax.transform.MetaScheduleTuneIRMod(
    #                 params={}, work_dir=work_dir, max_trials_global=2000
    #             ),
    #             relax.transform.MetaScheduleApplyDatabase(work_dir),
    #         ]
    #     )(mod)
    func = mod["compute"]
    database = ms.tir_integration.tune_tir(
        mod=func,
        target=target,
        work_dir=work_dir,
        max_trials_global=2000,
    )
    # database = JSONDatabase(work_dir=work_dir)
    sch = ms.tir_integration.compile_tir(database, func, target)
    print(sch.trace, file=open(trace_path, "w"))
    mod["compute"] = sch.mod["main"]

mod = LiftTIRGlobalBufferAlloc()(mod)
print(mod.script(), file=open(after_path, "w"))
print("<schedule done>")


# build
# func = next(mod.functions.values())
with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
    ex = relax.build(mod, target=target)
if target.kind.name != "llvm":
    print(ex.mod.imported_modules[0].imported_modules[0].get_source(), file=open(dump_path, "w"))
ex.export_library(ex_path)
vm = relax.VirtualMachine(ex, dev, profile=True)


atol, rtol = 1e-3, 1e-3

inputs = [
    torch.randn(4096, 4096, dtype=torch.float16).to("mps"),
    torch.randn(2048, 4096, dtype=torch.float16).to("mps"),
    # torch.zeros(b, s, 11008, dtype=torch.float16).cuda(),
]
tvm_inputs = [tvm.nd.array(x.detach().cpu().numpy(), dev) for x in inputs]
tvm_res = vm["main"](*tvm_inputs)
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

torch_res_0 = (inputs[1].to(torch.float32) @ inputs[0].to(torch.float32)).to(torch.float16)


close = np.allclose(torch_res_0.detach().cpu().numpy(), tvm_res.numpy(), atol=atol, rtol=rtol)
if not close:
    print("torch:\n", torch_res_0.detach().cpu().numpy())
    print("tvm:\n", tvm_res.numpy())
    assert close

print("<correctness check done>")


flush_cache()
torch.mps.synchronize()
report = vm.profile("main", *tvm_inputs)
print(report)

operator_call, operator_tm = None, None
for op in report.calls:
    if operator_call is None or op["Duration (us)"].microseconds > operator_tm:
        operator_call, operator_tm = op, op["Duration (us)"].microseconds
print(operator_call)


tflops = 2048 * 4096 * 4096 * 2 / operator_tm / 1e6
print(f"Op latency: {operator_tm} us, TFlops: {tflops}")


flush_cache()
torch.mps.synchronize()
time = monotonic()
torch_res_0 = (inputs[1].to(torch.float32) @ inputs[0].to(torch.float32)).to(torch.float16)
torch.mps.synchronize()
time = monotonic() - time

tflops = 2048 * 4096 * 4096 * 2 / time / 1e12
print(f"Torch Op latency: {time * 1e6} us, TFlops: {tflops}")


print("<performance check done>")
