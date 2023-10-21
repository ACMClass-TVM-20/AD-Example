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


@I.ir_module
class Module:
    @T.prim_func
    def main(lv37: T.Buffer((T.int64(4096), T.int64(1376)), "uint32"), lv38: T.Buffer((T.int64(4096), T.int64(344)), "float16"), p_lv105: T.handle, p_lv41: T.handle, p_lv26: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv105 = T.match_buffer(p_lv105, (T.int64(1), n, T.int64(11008)), "float16")
        lv41 = T.match_buffer(p_lv41, (T.int64(1), n, T.int64(4096)), "float16")
        lv26 = T.match_buffer(p_lv26, (T.int64(1), n, T.int64(4096)), "float16")
        p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)), "float16")
        # with T.block("root"):
        p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)))
        var_compute_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
        for i, j in T.grid(T.int64(4096), T.int64(11008)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv37[v_i, v_j // T.int64(8)], lv38[v_i, v_j // T.int64(32)])
                T.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv37[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv38[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(11008)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv105[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv105[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_i2, v_k])
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
                var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_compute_intermediate[v_ax0, v_ax1, v_ax2], lv41[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate[v_ax0, v_ax1, v_ax2] + lv41[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv26[v_ax0, v_ax1, v_ax2], var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv26[v_ax0, v_ax1, v_ax2] + var_T_add_intermediate[v_ax0, v_ax1, v_ax2]


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

print(mod.script(), file=open(after_path, "w"))
print("<schedule done>")

# build
func = next(mod.functions.values())
with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
    ex = tvm.build(func, target=target)
if target.kind.name == "cuda":
    print(ex.imported_modules[0].get_source(), file=open(dump_path, "w"))
ex.export_library(ex_path)
    # @T.prim_func
    # def main(lv37: T.Buffer((T.int64(4096), T.int64(1376)), "uint32"), lv38: T.Buffer((T.int64(4096), T.int64(344)), "float16"), p_lv105: T.handle, p_lv41: T.handle, p_lv26: T.handle, p_output0: T.handle):
    #     T.func_attr({"tir.noalias": T.bool(True)})
    #     n = T.int64()
    #     lv105 = T.match_buffer(p_lv105, (T.int64(1), n, T.int64(11008)), "float16")
    #     lv41 = T.match_buffer(p_lv41, (T.int64(1), n, T.int64(4096)), "float16")
    #     lv26 = T.match_buffer(p_lv26, (T.int64(1), n, T.int64(4096)), "float16")
    #     p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)), "float16")
    #     # with T.block("root"):
    #     p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    #     var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)))
    #     var_compute_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
    #     var_T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")

inputs = [torch.randn(4096, 11008, dtype=torch.float16), torch.randn(1, 11008, )]
ex(tvm_inputs_cp[0], tvm_inputs_cp[1], tvm_inputs_cp[2])
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






# configs
check_correctness, check_performance, check_register_usage = True, True, False

# batch, shape_m, shape_k, shape_n = 1, 4096, 4096, 4096
batch, shape_m, shape_k, shape_n = 4, 511, 4096, 4094
transpose_A, transpose_B = False, False

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
shape_2 = (shape_k, shape_n)
shape_3 = (batch, shape_m, shape_n)
tvm_shape_1 = (tvm_batch, tvm_shape_m, tvm_shape_k)
tvm_shape_2 = (tvm_shape_k, tvm_shape_n)
tvm_shape_3 = (tvm_batch, tvm_shape_m, tvm_shape_n)

if transpose_A:
    shape_1[1],

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
    tvm_inputs_cp = tvm_inputs
    # tvm_inputs_cp = list(tvm_inputs)
    # if transpose_A is False and transpose_B is True:
    #     # swap a and b because the place of the params are swapped by FuseOpsByPattern pass
    #     tvm_inputs_cp[0], tvm_inputs_cp[1] = tvm_inputs_cp[1], tvm_inputs_cp[0]

    ex(tvm_inputs_cp[0], tvm_inputs_cp[1], tvm_inputs_cp[2])
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

    tvm_inputs_cp = list(tvm_inputs)
    if transpose_A is False and transpose_B is True:
        # swap a and b because the place of the params are swapped by FuseOpsByPattern pass
        tvm_inputs_cp[0], tvm_inputs_cp[1] = tvm_inputs_cp[1], tvm_inputs_cp[0]

    report = eval(tvm_inputs_cp[0], tvm_inputs_cp[1], tvm_inputs_cp[2])
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
