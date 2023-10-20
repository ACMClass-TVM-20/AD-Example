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


# configs
check_correctness, check_performance, check_register_usage = True, True, False

# batch, shape_m, shape_k, shape_n = 1, 4096, 4096, 4096
batch, shape_m, shape_k, shape_n = 4, 512, 4096, 4096
transpose_A, transpose_B = False, True

if len(sys.argv) > 1:
    batch, shape_m, shape_k, shape_n = [int(x) for x in sys.argv[1:5]]

dtype, fallback_dtype, shape_dtype = "float16", "float32", "int64"
atol, rtol = 1e-3, 1e-3

print(f"Running with dtype, fallback_dtype, shape_dtype = {dtype, fallback_dtype, shape_dtype}")
print(f"Running with batch, shape_m, shape_k, shape_n = {batch, shape_m, shape_k, shape_n}")
print(f"Running with transpose_A, transpose_B = {transpose_A, transpose_B}")
print(f"Running with atol, rtol = {atol, rtol}")


# handle shapes
def handle_symbolic_shape(val, name):
    return (val, val) if val > 0 else (-val, tir.Var(name, shape_dtype))


batch, tvm_batch = handle_symbolic_shape(batch, "batch")
shape_m, tvm_shape_m = handle_symbolic_shape(shape_m, "m")
shape_k, tvm_shape_k = handle_symbolic_shape(shape_k, "k")
shape_n, tvm_shape_n = handle_symbolic_shape(shape_n, "n")

shape_1 = (batch, shape_m, shape_k)
shape_2 = (shape_n, shape_k)
shape_3 = (batch, shape_m, shape_n)
tvm_shape_1 = (tvm_batch, tvm_shape_m, tvm_shape_k)
tvm_shape_2 = (tvm_shape_n, tvm_shape_k)
tvm_shape_3 = (tvm_batch, tvm_shape_m, tvm_shape_n)


# devices and paths
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
    def transpose_if(var, cond):
        return R.permute_dims(var) if cond else var

    A = relax.Var("A", relax.TensorStructInfo(tvm_shape_1, dtype))
    B = relax.Var("B", relax.TensorStructInfo(tvm_shape_2, dtype))
    bb = BlockBuilder()
    with bb.function("main", [A, B]):
        with bb.dataflow():
            lv = bb.emit(
                relax.op.matmul(
                    transpose_if(A, transpose_A),
                    transpose_if(B, transpose_B),
                    out_dtype=fallback_dtype,
                )
            )
            gv = bb.emit_output(relax.op.astype(lv, dtype))
        bb.emit_func_output(gv)

    mod = bb.get()
    mod.show(None, False)
    return mod


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
    mod.show()
    exit()
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.FuseTIR()(mod)
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    mod = relax.transform.FuseOps()(mod)
    mod = relax.transform.FuseTIR()(mod)

    print(mod.script(), file=open(before_path, "w"))
    print("<transform done>")
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


# Step 1. check correctness
def fn_check_correctness(
    ex: tvm.runtime.Module, tvm_inputs: List[tvm.runtime.NDArray], torch_inputs: List[torch.Tensor]
):
    ex(tvm_inputs[1], tvm_inputs[0], tvm_inputs[2])
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch_res = (torch_inputs[0].T if transpose_A else torch_inputs[0]) @ (
        torch_inputs[1].T if transpose_B else torch_inputs[1]
    )

    close = np.allclose(
        torch_res.detach().cpu().numpy(), tvm_inputs[2].numpy(), atol=atol, rtol=rtol
    )
    if not close:
        print("torch:\n", torch_res.detach().cpu().numpy())
        print("tvm:\n", tvm_inputs[2].numpy())
        assert close

    print("<correctness check done>")


# Step 2. check performance
def fn_check_performance(ex: tvm.runtime.Module, tvm_inputs: List[tvm.runtime.NDArray]):
    eval = ex.time_evaluator(ex.entry_name, dev, 10, 10)
    report = eval(tvm_inputs[1], tvm_inputs[0], tvm_inputs[2])
    print(report)

    op_time = report.mean
    tflops = batch * shape_m * shape_n * shape_k * 2 / op_time / 1e12
    print(f"Op latency: {op_time*1e6} us, TFlops: {tflops}")
    print("<performance check done>")


# Step 3. check register usage
def fn_check_register_usage():
    os.system(
        f"nvcc -maxrregcount=255 -arch=sm_89 --cubin -w -Xptxas -v {dump_path} -o {cubin_path}"
    )
    print("<register usage check done>")


if __name__ == "__main__":
    # generate and build module
    mod = get_mod()
    mod = transform_mod(mod)
    ex = build_mod(mod)

    # generate inputs
    np_inputs = [np.random.normal(size=size).astype(dtype) for size in [shape_1, shape_2, shape_3]]
    torch_inputs = [torch.tensor(x).to("cuda") for x in np_inputs[:2]]
    tvm_inputs = [tvm.nd.array(x, dev) for x in np_inputs]

    # run checks
    if check_correctness:
        fn_check_correctness(ex, tvm_inputs, torch_inputs)

    if check_performance:
        fn_check_performance(ex, tvm_inputs)

    if check_register_usage:
        fn_check_register_usage()
