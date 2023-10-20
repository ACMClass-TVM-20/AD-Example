"""Test: fp16 + decode
"""
import os
import sys

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
from fuse_dequantize_matmul import decode_matmul_pattern
from dequantize import q3f16_1, q4f16_1

batch, shape_m, shape_k, shape_n = 1, 4094, 4096, 4096
quantize_mode = q4f16_1

if len(sys.argv) > 1:
    batch, shape_m, shape_k, shape_n = [int(x) for x in sys.argv[1:5]]

shape_1 = (batch, shape_m, shape_k)
shape_2 = (shape_n, shape_k)
shape_3 = (batch, shape_m, shape_n)
dtype = "float16"
fallback_dtype = "float32"
atol = 1e-5
rtol = 1e-5

check_correctness, check_performance, check_register_usage = True, True, True

print(f"Running with dtype={dtype}, fallback_dtype={fallback_dtype}")
print(f"Running with batch, shape_m, shape_k, shape_n = {batch, shape_m, shape_k, shape_n}")

# target, dev = tvm.target.Target("llvm"), tvm.cpu()
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


def get_mod():
    bb = BlockBuilder()
    A = relax.Var("A", relax.TensorStructInfo(shape_1, dtype))
    B = relax.Var("B", relax.TensorStructInfo(shape_2, dtype))
    quantized_sinfo = quantize_mode.get_dequantize_sinfo(B.struct_info)
    quantized = [relax.Var("weight", quantized_sinfo[0]), relax.Var("scale", quantized_sinfo[1])]
    with bb.function("main", [A, *quantized]):
        with bb.dataflow():
            B_dequantize = bb.emit_te(
                quantize_mode.get_dequantize_func(B.struct_info),
                *quantized,
                primfunc_name_hint="dequantize",
            )
            lv1 = bb.emit(relax.op.permute_dims(B_dequantize))
            lv2 = bb.emit(relax.op.matmul(A, lv1, out_dtype=fallback_dtype))
            gv = bb.emit_output(relax.op.astype(lv2, dtype))
        bb.emit_func_output(gv)
    Module = bb.get()
    return Module


def transform_mod(mod):
    def transpose_matmul_pattern():
        w = wildcard()
        x = wildcard()
        wT = is_op("relax.permute_dims")(w)
        o = is_op("relax.matmul")(x, wT)
        annotations = {"o": o, "w": w, "x": x, "wT": wT}

        def _check(context: relax.transform.PatternCheckContext) -> bool:
            transpose_call = context.annotated_expr["wT"]
            ndim = transpose_call.args[0].struct_info.ndim
            if ndim == -1:
                return False
            if ndim == 2 and transpose_call.attrs.axes is None:
                return True
            axes = list(range(ndim))
            axes[-1], axes[-2] = axes[-2], axes[-1]
            return list(transpose_call.attrs.axes) == axes

        return o, annotations, _check

    mod = relax.transform.FuseOpsByPattern(
        [("transpose_matmul_fuse", *transpose_matmul_pattern())]
    )(mod)
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.FuseT)
    return mod


def build_mod(mod):
    if target.kind.name == "cuda":
        with target, tvm.transform.PassContext(trace=Trace(mod)):
            mod = dl.ApplyDefaultSchedule(dl.gpu.matmul.MatmulTensorizationMMA())(mod)
            # mod = dl.ApplyDefaultSchedule(dl.gpu.matmul.Matmul())(mod)

    print(mod.script(), file=open(after_path, "w"))
    print("<schedule done>")

    # build
    func = next(mod.functions.values())
    with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
        ex = tvm.build(func, target=target)
    if target.kind.name == "cuda":
        print(ex.imported_modules[0].get_source(), file=open(dump_path, "w"))
    ex.export_library(ex_path)
    print("<build done>")
    return ex
IR()(mod)
    mod = relax.transform.FuseOpsByPattern([("decode_matmul", *decode_matmul_pattern())])(mod)
    mod = relax.transform.FuseTIR()(mod)
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    mod = relax.transform.FuseOps()(mod)
    mod = relax.transform.FuseTIR()(mod)

    print(mod.script(), file=open(before_path, "w"))
    print("<transform done>"

mod = get_mod()
mod = transform_mod(mod)
ex = build_mod(mod)


# generate inputs
np_inputs = [np.random.normal(size=size).astype(dtype) for size in [shape_1, shape_2, shape_3]]
torch_inputs = [torch.tensor(x).to("cuda") for x in np_inputs[:2]]
tvm_inputs = [tvm.nd.array(x, dev) for x in np_inputs]


# Step 0: quantize


# Step 1. check correctness
if check_correctness:
    ex(tvm_inputs[1], tvm_inputs[0], tvm_inputs[2])
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch_res = torch_inputs[0] @ torch_inputs[1].T
    print("torch:\n", torch_res.detach().cpu().numpy())
    print("tvm:\n", tvm_inputs[2].numpy())
    assert np.allclose(
        torch_res.detach().cpu().numpy(), tvm_inputs[2].numpy(), atol=1e-3, rtol=1e-3
    )
    # assert np.allclose(torch_res.detach().cpu().numpy(), tvm_inputs[2].numpy(), atol=atol, rtol=rtol)
    print("<correctness check done>")

# Step 2. check performance
if check_performance:
    eval = ex.time_evaluator(ex.entry_name, dev, 10, 10)
    report = eval(tvm_inputs[1], tvm_inputs[0], tvm_inputs[2])
    print(report)

    op_time = report.mean
    tflops = batch * shape_m * shape_n * shape_k * 2 / op_time / 1e12
    print(f"Op latency: {op_time*1e6} us, TFlops: {tflops}")
    print("<performance check done>")

# Step 3. check register usage
if check_register_usage:
    os.system(
        f"nvcc -maxrregcount=255 -arch=sm_89 --cubin -w -Xptxas -v {dump_path} -o {cubin_path}"
    )
    print("<register usage check done>")
