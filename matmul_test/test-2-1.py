"""Test: fp16 mixed precision matmul + epilogue (mul2)
- With unroll: 98.49008892671615
- Without unroll: 144.54291252099756
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

reload = False
batch, shape_m, shape_k, shape_n = 6, 512, 4096, 11008

if len(sys.argv) > 1:
    batch, shape_m, shape_k, shape_n = [int(x) for x in sys.argv[1:5]]

print(f"Running with batch, shape_m, shape_k, shape_n = {batch, shape_m, shape_k, shape_n}")

shape_1 = (batch, shape_m, shape_k)
shape_2 = (shape_n, shape_k)
shape_3 = (batch, shape_m, shape_n)
dtype = "float16"
fallback_dtype = "float32"
atol = 1e-5
rtol = 1e-5

# target, dev = tvm.target.Target("llvm"), tvm.cpu()
target, dev = tvm.target.Target("nvidia/geforce-rtx-4090"), tvm.cuda()

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

if not reload:

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor(shape_1, dtype), B: R.Tensor(shape_2, dtype)):
            with R.dataflow():
                lv1 = R.permute_dims(B)
                lv2 = R.matmul(A, lv1, out_dtype=fallback_dtype)
                lv3 = R.astype(lv2, dtype)
                gv = lv3 * R.const(2, dtype)
                R.output(gv)
            return gv

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

    mod = Module
    mod = relax.transform.FuseOpsByPattern(
        [("transpose_matmul_fuse", *transpose_matmul_pattern())]
    )(mod)
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.FuseTIR()(mod)
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    mod = relax.transform.FuseOps()(mod)
    mod = relax.transform.FuseTIR()(mod)

    print(mod.script(), file=open(before_path, "w"))
    print("<transform done>")
    if target.kind.name == "cuda":
        with target, tvm.transform.PassContext(trace=Trace(mod)):
            mod = dl.ApplyDefaultSchedule(dl.gpu.matmul.MatmulTensorization())(mod)
            # mod = dl.ApplyDefaultSchedule(dl.gpu.matmul.Matmul())(mod)

    print(mod.script(), file=open(after_path, "w"))
    print("<schedule done>")

    # build
    func_name = "fused_fused_relax_permute_dims_relax_matmul_cast_multiply"
    with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
        ex = tvm.build(mod[func_name], target=target)
    if target.kind.name == "cuda":
        print(ex.imported_modules[0].get_source(), file=open(dump_path, "w"))
    ex.export_library(ex_path)
    print("<build done>")
else:
    ex = tvm.runtime.load_module(ex_path)
    print("<reload done>")


# generate inputs
np_inputs = [np.random.normal(size=size).astype(dtype) for size in [shape_1, shape_2, shape_3]]
torch_inputs = [torch.tensor(x).to("cuda") for x in np_inputs[:2]]
tvm_inputs = [tvm.nd.array(x, dev) for x in np_inputs]

# Step 1. check correctness
ex(tvm_inputs[1], tvm_inputs[0], tvm_inputs[2])
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch_res = torch_inputs[0] @ torch_inputs[1].T * 2
assert np.allclose(torch_res.detach().cpu().numpy(), tvm_inputs[2].numpy(), atol=1e-3, rtol=1e-3)
# assert np.allclose(torch_res.detach().cpu().numpy(), tvm_inputs[2].numpy(), atol=atol, rtol=rtol)
print("<correctness check done>")

# Step 2. check performance
eval = ex.time_evaluator(ex.entry_name, dev, 10, 10)
report = eval(tvm_inputs[1], tvm_inputs[0], tvm_inputs[2])
print(report)

op_time = report.mean
tflops = (batch * shape_m * shape_n * shape_k * 2 + batch * shape_m * shape_n) / op_time / 1e12
print(f"Op latency: {op_time*1e6} us, TFlops: {tflops}")
print("<performance check done>")

# Step 3. check register usage
os.system(f"nvcc -maxrregcount=255 -arch=sm_89 --cubin -w -Xptxas -v {dump_path} -o {cubin_path}")
print("<register usage check done>")
