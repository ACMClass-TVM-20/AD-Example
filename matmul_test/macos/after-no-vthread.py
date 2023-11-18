# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def compute(B: T.Buffer((T.int64(4096), T.int64(4096)), "float16"), p_A: T.handle, p_O: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        b = T.int64()
        A = T.match_buffer(p_A, (b, T.int64(512), T.int64(4096)), "float16")
        O = T.match_buffer(p_O, (b, T.int64(512), T.int64(4096)), "float16")
        # with T.block("root"):
        O_intermediate_reindex_local = T.alloc_buffer((T.int64(1), b * T.int64(512), T.int64(4096)), scope="local")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(128), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(b * T.int64(16), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                for ax2_3_init, ax1_3_0_init in T.grid(T.int64(4), T.int64(4)):
                                    for ax1_3_1_init in T.vectorized(T.int64(1)):
                                        with T.block("matmul_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(b * T.int64(512), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_0_init + ax1_3_1_init)
                                            v2 = T.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(32) + ax2_1 * T.int64(32) + ax2_2 * T.int64(4) + ax2_3_init)
                                            T.reads()
                                            T.writes(O_intermediate_reindex_local[T.int64(0), v1, v2])
                                            O_intermediate_reindex_local[T.int64(0), v1, v2] = T.float32(0)
                                for ax3_0, ax3_1, ax2_3, ax1_3_0 in T.grid(T.int64(512), T.int64(8), T.int64(4), T.int64(4)):
                                    for ax1_3_1 in T.vectorized(T.int64(1)):
                                        with T.block("matmul_update"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(b * T.int64(512), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_0 + ax1_3_1)
                                            v2 = T.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(32) + ax2_1 * T.int64(32) + ax2_2 * T.int64(4) + ax2_3)
                                            v3 = T.axis.reduce(T.int64(4096), ax3_0 * T.int64(8) + ax3_1)
                                            T.reads(O_intermediate_reindex_local[T.int64(0), v1, v2], A[v1 // T.int64(512), v1 % T.int64(512), v3], B[v3, v2])
                                            T.writes(O_intermediate_reindex_local[T.int64(0), v1, v2])
                                            O_intermediate_reindex_local[T.int64(0), v1, v2] = O_intermediate_reindex_local[T.int64(0), v1, v2] + T.Cast("float32", A[v1 // T.int64(512), v1 % T.int64(512), v3]) * T.Cast("float32", B[v3, v2])
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(4)):
                                    for ax2_1_1 in T.vectorized(T.int64(1)):
                                        with T.block("O_intermediate_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(b * T.int64(512), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(32) + ax2_2 * T.int64(4) + ax2_0 + ax2_1_1)
                                            T.reads(O_intermediate_reindex_local[v0, v1, v2])
                                            T.writes(O[v1 // T.int64(512), v1 % T.int64(512), v2])
                                            if v1 // T.int64(512) < b:
                                                O[v1 // T.int64(512), v1 % T.int64(512), v2] = T.Cast("float16", O_intermediate_reindex_local[v0, v1, v2])

    @R.function
    def main(w: R.Tensor((4096, 4096), dtype="float16"), x: R.Tensor(("b", 512, 4096), dtype="float16")) -> R.Tensor(("b", 512, 4096), dtype="float16"):
        b = T.int64()
        cls = Module
        out = R.call_tir(cls.compute, (w, x), out_sinfo=R.Tensor((b, 512, 4096), dtype="float16"))
        return out
