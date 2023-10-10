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
import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import te
from tvm.testing.tir import mma_schedule
from tvm.tir.tensor_intrin.cuda import (
    LDMATRIX_16x16_A_INTRIN,
    LDMATRIX_16x16_B_INTRIN,
    LDMATRIX_16x16_B_TRANS_INTRIN,
    LDMATRIX_16x32_A_INTRIN,
    LDMATRIX_16x32_B_TRANS_INTRIN,
    LDMATRIX_32x16_B_INTRIN,
    MMA_f16f16f16_INTRIN,
    MMA_f16f16f16_TRANS_INTRIN,
    MMA_f16f16f32_INTRIN,
    MMA_f16f16f32_TRANS_INTRIN,
    MMA_fill_16x16_f16_INTRIN,
    MMA_fill_16x16_f32_INTRIN,
    MMA_fill_16x16_i32_INTRIN,
    MMA_i8i8i32_INTRIN,
    MMA_i8i8i32_TRANS_INTRIN,
    MMA_store_16x16_f16_global_INTRIN,
    MMA_store_16x16_f32_global_INTRIN,
    MMA_store_16x16_i32_global_INTRIN,
    shared_16x16_to_ldmatrix_32x8_layout,
    shared_16x32_to_ldmatrix_32x16_layout,
    shared_32x16_to_ldmatrix_32x16_layout,
)


def mma_schedule(
    workload,
    k_inner,
    in_dtype,
    b_transposed,
    i_factors,
    j_factors,
    k_factors,
    index_map_A,
    index_map_B,
    index_map_C,
    ldmatrix_a_intrin,
    ldmatrix_b_intrin,
    mma_intrin,
    mma_fill_intrin,
    mma_store_intrin,
    shared_scope="shared",
):
    """Create a tensorized schedule for GEMM with MMA intrinsics."""
    import tvm  # pylint: disable=import-outside-toplevel

    ir_module = tvm.IRModule({"main": workload})
    sch = tvm.tir.Schedule(ir_module)

    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    i, i_tc = sch.split(i, factors=[None, 16])
    j, j_tc = sch.split(j, factors=[None, 16])
    k, k_tc = sch.split(k, factors=[None, k_inner])

    sch.reorder(i, j, k, i_tc, j_tc, k_tc)

    block_inner = sch.blockize(i_tc)
    block_outer, block_inner = block_inner, block

    num_ty = i_factors[2] * j_factors[2]

    i0, i1, i2, i3, i4 = sch.split(i, factors=i_factors)
    j0, j1, j2, j3, j4 = sch.split(j, factors=j_factors)
    k0, k1, k2 = sch.split(k, k_factors)

    sch.reorder(i0, j0, i1, j1, j2, i2, k0, k1, i3, j3, k2, i4, j4)

    block_idx = sch.fuse(i0, j0)
    block_idy = sch.fuse(i1, j1)
    thread_idy = sch.fuse(j2, i2)
    sch.bind(block_idx, "blockIdx.x")
    sch.bind(block_idy, "blockIdx.y")
    sch.bind(thread_idy, "threadIdx.y")

    def fetch_to_shared(block, idx, ndim):
        block_read = sch.cache_read(block, idx, shared_scope)
        sch.compute_at(block_read, k0)
        vector_size = 16 if in_dtype == "int8" else 8
        warp_size = 32
        fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])
        _, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])
        sch.bind(f_2, "threadIdx.x")
        sch.bind(f_1, "threadIdx.y")
        sch.vectorize(f_3)
        offset = 8 if in_dtype == "float16" else 16
        sch.storage_align(block_read, 0, axis=-2, factor=32, offset=offset)

        return block_read

    fetch_to_shared(block_outer, 0, 2)
    fetch_to_shared(block_outer, 1, 2)

    A_warp = sch.cache_read(block_outer, 0, "warp")
    B_warp = sch.cache_read(block_outer, 1, "warp")

    sch.compute_at(A_warp, k1)
    sch.compute_at(B_warp, k1)

    C_warp = sch.cache_write(block_outer, 0, "warp")
    sch.reverse_compute_at(C_warp, thread_idy)

    ii, jj = sch.get_loops(C_warp)[-2:]
    io, ii = sch.split(ii, factors=[None, 16])
    jo, ji = sch.split(jj, factors=[None, 16])
    sch.reorder(io, jo, ii, ji)

    sch.decompose_reduction(block_outer, sch.get_loops(block_outer)[3])
    block_init_c = sch.get_block("C_init")

    def tile_wmma_fragment(block_read, height, width):
        i, j = sch.get_loops(block_read)[-2:]
        i0, i1 = sch.split(i, factors=[None, height])
        j0, j1 = sch.split(j, factors=[None, width])
        sch.reorder(i0, j0, i1, j1)
        return i1

    loop_a = tile_wmma_fragment(A_warp, 16, k_inner)

    if b_transposed:
        loop_b = tile_wmma_fragment(B_warp, 16, k_inner)
    else:
        loop_b = tile_wmma_fragment(B_warp, k_inner, 16)

    sch.transform_layout(A_warp, ("write", 0), index_map_A)
    sch.transform_layout(B_warp, ("write", 0), index_map_B)
    sch.transform_layout(C_warp, ("read", 0), index_map_C)

    sch.tensorize(loop_a, ldmatrix_a_intrin)
    sch.tensorize(loop_b, ldmatrix_b_intrin)
    sch.tensorize(sch.get_loops(block_inner)[-3], mma_intrin)
    sch.tensorize(sch.get_loops(block_init_c)[-2], mma_fill_intrin)
    sch.tensorize(sch.get_loops(C_warp)[-2], mma_store_intrin)

    return sch


M = 4096
N = 4096
K = 4096
measure_perf = True
gflops = (N * M * K) * 2 / 1e9


def matmul(m, n, k, in_dtype, out_dtype, b_transposed):
    b_shape = (n, k) if b_transposed else (k, n)
    a = te.placeholder((m, k), name="A", dtype=in_dtype)
    b = te.placeholder(b_shape, name="B", dtype=in_dtype)
    k = te.reduce_axis((0, k), name="k")

    def maybe_cast(v):
        if in_dtype != out_dtype:
            return tvm.tir.Cast(out_dtype, v)
        return v

    def maybe_swap(i, j):
        if b_transposed:
            return j, i
        return i, j

    c = te.compute(
        (m, n),
        lambda i, j: te.sum(maybe_cast(a[i, k]) * maybe_cast(b[maybe_swap(k, j)]), axis=[k]),
        name="C",
    )
    return (a, b, c)


def run_test(
    k_inner,
    in_dtype,
    out_dtype,
    b_transposed,
    i_factors,
    j_factors,
    k_factors,
    index_map_A,
    index_map_B,
    index_map_C,
    ldmatrix_a_intrin,
    ldmatrix_b_intrin,
    mma_intrin,
    mma_fill_intrin,
    mma_store_intrin,
):
    sch = mma_schedule(
        te.create_prim_func(matmul(M, N, K, in_dtype, out_dtype, b_transposed)),
        k_inner,
        in_dtype,
        b_transposed,
        i_factors,
        j_factors,
        k_factors,
        index_map_A,
        index_map_B,
        index_map_C,
        ldmatrix_a_intrin,
        ldmatrix_b_intrin,
        mma_intrin,
        mma_fill_intrin,
        mma_store_intrin,
    )

    f = tvm.build(sch.mod["main"], target="cuda", name="dense")

    dev = tvm.device("cuda", 0)

    if in_dtype == "float16":
        a_np = np.random.uniform(size=(M, K)).astype("float16")

        if b_transposed:
            b_np = np.random.uniform(size=(N, K)).astype("float16")
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32").transpose()).astype(
                out_dtype
            )
        else:
            b_np = np.random.uniform(size=(K, N)).astype("float16")
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32")).astype(out_dtype)
    else:
        a_np = np.random.randint(-128, 128, (M, K)).astype("int8")

        if b_transposed:
            b_np = np.random.randint(-128, 128, (N, K)).astype("int8")
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32").transpose()).astype(
                "int32"
            )
        else:
            b_np = np.random.randint(-128, 128, (K, N)).astype("int8")
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32")).astype("int32")

    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((M, N), dtype=out_dtype), dev)

    f(a, b, c)

    if out_dtype != "float16":
        # The numpy reference is computed with fp32 precision (otherwise too slow).
        # So there is non-trivial accuracy difference if TVM result is computed with fp16 accumulation.
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    return lambda: f.time_evaluator(f.entry_name, dev, number=500)(a, b, c)


@tvm.testing.requires_cuda_compute_version(8)
def test_f16f16f32_m16n16k16():
    def index_map(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_ldmatrix_32x8_layout(i % 16, j % 16),
        )

    k_inner = 16
    in_dtype = "float16"
    out_dtype = "float32"
    i_factors, j_factors, k_factors = [4, 8, 2, 4, 1], [1, 64, 2, 1, 2], [128, 2, 1]

    # timer = run_test(
    #     k_inner,
    #     in_dtype,
    #     out_dtype,
    #     False,  # b_transposed
    #     i_factors,
    #     j_factors,
    #     k_factors,
    #     index_map,
    #     index_map,
    #     index_map,
    #     LDMATRIX_16x16_A_INTRIN,
    #     LDMATRIX_16x16_B_INTRIN,
    #     MMA_f16f16f32_INTRIN,
    #     MMA_fill_16x16_f32_INTRIN,
    #     MMA_store_16x16_f32_global_INTRIN,
    # )

    # if measure_perf and timer:
    #     print("f16f16f32_m16n16k16: %f GFLOPS" % (gflops / (timer().mean)))

    timer = run_test(
        k_inner,
        in_dtype,
        out_dtype,
        True,  # b_transposed
        i_factors,
        j_factors,
        k_factors,
        index_map,
        index_map,
        index_map,
        LDMATRIX_16x16_A_INTRIN,
        LDMATRIX_16x16_B_TRANS_INTRIN,
        MMA_f16f16f32_TRANS_INTRIN,
        MMA_fill_16x16_f32_INTRIN,
        MMA_store_16x16_f32_global_INTRIN,
    )

    if measure_perf and timer:
        print("f16f16f32_m16n16k16_trans: %f GFLOPS" % (gflops / (timer().mean)))
test_f16f16f32_m16n16k16()



reload = False
batch, shape_m, shape_k, shape_n = 1, 4096, 4096, 4096

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
target, dev = tvm.target.Target("nvidia/geforce-rtx-4090"), tvm.cuda()
# target, dev = tvm.target.Target("nvidia/nvidia-a100"), tvm.cuda()

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
                gv = R.matmul(A, lv1, out_dtype=fallback_dtype)
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
            mod = dl.ApplyDefaultSchedule(dl.gpu.matmul.MatmulTensorizationMMA())(mod)
            # mod = dl.ApplyDefaultSchedule(dl.gpu.matmul.Matmul())(mod)

    print(mod.script(), file=open(after_path, "w"))
    print("<schedule done>")

    # build
    func_name = "fused_relax_permute_dims_relax_matmul"
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
np_inputs = [np.random.normal(size=size).astype(dtype) for size in [shape_1, shape_2]] + [np.random.normal(size=shape_3).astype(fallback_dtype)]
torch_inputs = [torch.tensor(x).to("cuda") for x in np_inputs[:2]]
tvm_inputs = [tvm.nd.array(x, dev) for x in np_inputs]

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
