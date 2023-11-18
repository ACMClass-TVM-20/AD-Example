# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def compute(B: T.Buffer((T.int64(4096), T.int64(4096)), "float16"), A: T.Buffer((T.int64(2048), T.int64(4096)), "float16"), O: T.Buffer((T.int64(2048), T.int64(4096)), "float16")):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        O_intermediate_local = T.alloc_buffer((T.int64(2048), T.int64(4096)), scope="local")
        A_shared = T.alloc_buffer((T.int64(2048), T.int64(4096)), "float16", scope="shared")
        B_shared = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16", scope="shared")
        for i0_0_i1_0_fused in T.thread_binding(T.int64(4096), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 512, "pragma_unroll_explicit": 1}):
            for i0_1_i1_1_fused in T.thread_binding(T.int64(1), thread="vthread.x"):
                for i0_2_i1_2_fused in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                    for i0_3_init, i1_3_init, i0_4_init, i1_4_init in T.grid(T.int64(8), T.int64(1), T.int64(1), T.int64(4)):
                        with T.block("matmul_init"):
                            v_i0 = T.axis.spatial(T.int64(2048), i0_0_i1_0_fused // T.int64(64) * T.int64(32) + i0_2_i1_2_fused // T.int64(16) * T.int64(8) + i0_3_init + i0_4_init)
                            v_i1 = T.axis.spatial(T.int64(4096), i0_0_i1_0_fused % T.int64(64) * T.int64(64) + i0_2_i1_2_fused % T.int64(16) * T.int64(4) + i1_3_init * T.int64(4) + i1_4_init)
                            T.reads()
                            T.writes(O_intermediate_local[v_i0, v_i1])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                            O_intermediate_local[v_i0, v_i1] = T.float32(0)
                    for k_0 in range(T.int64(512)):
                        for ax0_ax1_fused_0 in range(T.int64(4)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                with T.block("A_shared"):
                                    v0 = T.axis.spatial(T.int64(2048), i0_0_i1_0_fused // T.int64(64) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(64) + ax0_ax1_fused_1) // T.int64(8))
                                    v1 = T.axis.spatial(T.int64(4096), k_0 * T.int64(8) + (ax0_ax1_fused_0 * T.int64(64) + ax0_ax1_fused_1) % T.int64(8))
                                    T.reads(A[v0, v1])
                                    T.writes(A_shared[v0, v1])
                                    A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in range(T.int64(2)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("B_shared"):
                                        v0 = T.axis.spatial(T.int64(4096), k_0 * T.int64(8) + (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) // T.int64(64))
                                        v1 = T.axis.spatial(T.int64(4096), i0_0_i1_0_fused % T.int64(64) * T.int64(64) + (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) % T.int64(64))
                                        T.reads(B[v0, v1])
                                        T.writes(B_shared[v0, v1])
                                        B_shared[v0, v1] = B[v0, v1]
                        for k_1, i0_3, i1_3, k_2, i0_4, i1_4 in T.grid(T.int64(1), T.int64(8), T.int64(1), T.int64(8), T.int64(1), T.int64(4)):
                            with T.block("matmul_update"):
                                v_i0 = T.axis.spatial(T.int64(2048), i0_0_i1_0_fused // T.int64(64) * T.int64(32) + i0_2_i1_2_fused // T.int64(16) * T.int64(8) + i0_3 + i0_4)
                                v_i1 = T.axis.spatial(T.int64(4096), i0_0_i1_0_fused % T.int64(64) * T.int64(64) + i0_2_i1_2_fused % T.int64(16) * T.int64(4) + i1_3 * T.int64(4) + i1_4)
                                v_k = T.axis.reduce(T.int64(4096), k_0 * T.int64(8) + k_1 * T.int64(8) + k_2)
                                T.reads(O_intermediate_local[v_i0, v_i1], A_shared[v_i0, v_k], B_shared[v_k, v_i1])
                                T.writes(O_intermediate_local[v_i0, v_i1])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                O_intermediate_local[v_i0, v_i1] = O_intermediate_local[v_i0, v_i1] + T.Cast("float32", A_shared[v_i0, v_k]) * T.Cast("float32", B_shared[v_k, v_i1])
                    for ax0, ax1 in T.grid(T.int64(8), T.int64(4)):
                        with T.block("O_intermediate_local"):
                            v0 = T.axis.spatial(T.int64(2048), i0_0_i1_0_fused // T.int64(64) * T.int64(32) + i0_2_i1_2_fused // T.int64(16) * T.int64(8) + ax0)
                            v1 = T.axis.spatial(T.int64(4096), i0_0_i1_0_fused % T.int64(64) * T.int64(64) + i0_2_i1_2_fused % T.int64(16) * T.int64(4) + ax1)
                            T.reads(O_intermediate_local[v0, v1])
                            T.writes(O[v0, v1])
                            O[v0, v1] = T.Cast("float16", O_intermediate_local[v0, v1])

    @R.function
    def main(w: R.Tensor((4096, 4096), dtype="float16"), x: R.Tensor((2048, 4096), dtype="float16")) -> R.Tensor((2048, 4096), dtype="float16"):
        cls = Module
        out = R.call_tir(cls.compute, (w, x), out_sinfo=R.Tensor((2048, 4096), dtype="float16"))
        return out
