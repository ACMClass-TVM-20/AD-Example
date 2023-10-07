# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_fused_relax_permute_dims_relax_matmul_cast(B: T.Buffer((T.int64(11008), T.int64(4096)), "float16"), A: T.Buffer((T.int64(6), T.int64(512), T.int64(4096)), "float16"), var_compute_intermediate: T.Buffer((T.int64(6), T.int64(512), T.int64(11008)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_reindex_shared_dyn = T.alloc_buffer((T.int64(24), T.int64(128), T.int64(128), T.int64(32)), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(24), T.int64(128), T.int64(2), T.int64(2), T.int64(4), T.int64(1), T.int64(32), T.int64(8)), "float16", scope="warp")
        var_T_transpose_intermediate_reindex_shared_dyn = T.alloc_buffer((T.int64(86), T.int64(128), T.int64(128), T.int64(32)), "float16", scope="shared.dyn")
        var_T_transpose_intermediate_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(86), T.int64(128), T.int64(2), T.int64(2), T.int64(4), T.int64(1), T.int64(32), T.int64(8)), "float16", scope="warp")
        p_output0_intermediate_reindex_shared_dyn = T.alloc_buffer((T.int64(24), T.int64(86), T.int64(128), T.int64(128)), scope="shared.dyn")
        p_output0_intermediate_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(24), T.int64(86), T.int64(2), T.int64(2), T.int64(4), T.int64(4), T.int64(32), T.int64(8)), scope="warp")
        for ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused in T.thread_binding(T.int64(2064), thread="blockIdx.x"):
            for ax1_0_2_init in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                for ax2_0_2_init in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                    for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(4), T.int64(4)):
                        with T.block("matmul_o_init"):
                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                            v1_o = T.axis.spatial(T.int64(192), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(86) * T.int64(8) + ax1_0_2_init * T.int64(4) + ax1_0_3_init)
                            v2_o = T.axis.spatial(T.int64(688), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(86) * T.int64(8) + ax2_0_2_init * T.int64(4) + ax2_0_3_init)
                            T.reads()
                            T.writes(p_output0_intermediate_reindex_shared_dyn_warp[v1_o // T.int64(8), v2_o // T.int64(8), v1_o // T.int64(4) - v1_o // T.int64(8) * T.int64(2), v2_o // T.int64(4) - v2_o // T.int64(8) * T.int64(2), v1_o % T.int64(4), v2_o % T.int64(4), T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                            for ax1_1, ax2_1 in T.grid(T.int64(16), T.int64(16)):
                                with T.block("matmul_init"):
                                    v1_i_init, v2_i_init = T.axis.remap("SS", [ax1_1, ax2_1])
                                    T.reads()
                                    T.writes(p_output0_intermediate_reindex_shared_dyn_warp[(v1_o * T.int64(16) + v1_i_init) // T.int64(128), (v2_o * T.int64(16) + v2_i_init) // T.int64(128), (v1_o * T.int64(16) + v1_i_init) % T.int64(128) // T.int64(64), (v2_o * T.int64(16) + v2_i_init) % T.int64(128) // T.int64(64), (v1_o * T.int64(16) + v1_i_init) % T.int64(64) // T.int64(16), (v2_o * T.int64(16) + v2_i_init) % T.int64(64) // T.int64(16), v1_i_init % T.int64(8) * T.int64(4) + v2_i_init % T.int64(8) // T.int64(2), v2_i_init % T.int64(16) // T.int64(8) * T.int64(4) + v1_i_init % T.int64(16) // T.int64(8) * T.int64(2) + v2_i_init % T.int64(2)])
                                    p_output0_intermediate_reindex_shared_dyn_warp[(v1_o * T.int64(16) + v1_i_init) // T.int64(128), (v2_o * T.int64(16) + v2_i_init) // T.int64(128), (v1_o * T.int64(16) + v1_i_init) % T.int64(128) // T.int64(64), (v2_o * T.int64(16) + v2_i_init) % T.int64(128) // T.int64(64), (v1_o * T.int64(16) + v1_i_init) % T.int64(64) // T.int64(16), (v2_o * T.int64(16) + v2_i_init) % T.int64(64) // T.int64(16), v1_i_init % T.int64(8) * T.int64(4) + v2_i_init % T.int64(8) // T.int64(2), v2_i_init % T.int64(16) // T.int64(8) * T.int64(4) + v1_i_init % T.int64(16) // T.int64(8) * T.int64(2) + v2_i_init % T.int64(2)] = T.float32(0)
            for ax3_0_0 in range(T.int64(128)):
                for ax0_ax1_fused_0 in range(T.int64(4)):
                    for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                        for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                            for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                    with T.block("A_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(3072), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(86) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(4096), ax3_0_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(32))
                                        T.reads(A[v1 // T.int64(512), v1 % T.int64(512), v2])
                                        T.writes(A_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(32), v1 % T.int64(128), v2 % T.int64(32)])
                                        A_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(32), v1 % T.int64(128), v2 % T.int64(32)] = A[v1 // T.int64(512), v1 % T.int64(512), v2]
                for ax0_ax1_fused_0 in range(T.int64(4)):
                    for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                        for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                            for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                    with T.block("var_T_transpose_intermediate_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(11008), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(86) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(4096), ax3_0_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(32))
                                        T.reads(B[v1, v2])
                                        T.writes(var_T_transpose_intermediate_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(32), v1 % T.int64(128), v2 % T.int64(32)])
                                        var_T_transpose_intermediate_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(32), v1 % T.int64(128), v2 % T.int64(32)] = B[v1, v2]
                for ax1_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax2_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax3_0_1 in range(T.int64(2)):
                            for ax0_0, ax0_1, ax1 in T.grid(T.int64(4), T.int64(16), T.int64(16)):
                                with T.block("A_reindex_shared.dyn_warp"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(T.int64(3072), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(86) * T.int64(128) + ax1_0_2 * T.int64(64) + ax0_0 * T.int64(16) + ax0_1)
                                    v2 = T.axis.spatial(T.int64(4096), ax3_0_0 * T.int64(32) + ax3_0_1 * T.int64(16) + ax1)
                                    T.reads(A_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(32), v1 % T.int64(128), v2 % T.int64(32)])
                                    T.writes(A_reindex_shared_dyn_warp[v1 // T.int64(128), v2 // T.int64(32), v1 % T.int64(128) // T.int64(64), v2 % T.int64(32) // T.int64(16), v1 % T.int64(64) // T.int64(16), T.int64(0), v1 % T.int64(8) * T.int64(4) + v2 % T.int64(8) // T.int64(2), v2 % T.int64(16) // T.int64(8) * T.int64(4) + v1 % T.int64(16) // T.int64(8) * T.int64(2) + v2 % T.int64(2)])
                                    A_reindex_shared_dyn_warp[v1 // T.int64(128), v2 // T.int64(32), v1 % T.int64(128) // T.int64(64), v2 % T.int64(32) // T.int64(16), v1 % T.int64(64) // T.int64(16), T.int64(0), v1 % T.int64(8) * T.int64(4) + v2 % T.int64(8) // T.int64(2), v2 % T.int64(16) // T.int64(8) * T.int64(4) + v1 % T.int64(16) // T.int64(8) * T.int64(2) + v2 % T.int64(2)] = A_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(32), v1 % T.int64(128), v2 % T.int64(32)]
                            for ax0_0, ax0_1, ax1 in T.grid(T.int64(4), T.int64(16), T.int64(16)):
                                with T.block("var_T_transpose_intermediate_reindex_shared.dyn_warp"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(T.int64(11008), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(86) * T.int64(128) + ax2_0_2 * T.int64(64) + ax0_0 * T.int64(16) + ax0_1)
                                    v2 = T.axis.spatial(T.int64(4096), ax3_0_0 * T.int64(32) + ax3_0_1 * T.int64(16) + ax1)
                                    T.reads(var_T_transpose_intermediate_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(32), v1 % T.int64(128), v2 % T.int64(32)])
                                    T.writes(var_T_transpose_intermediate_reindex_shared_dyn_warp[v1 // T.int64(128), v2 // T.int64(32), v1 % T.int64(128) // T.int64(64), v2 % T.int64(32) // T.int64(16), v1 % T.int64(64) // T.int64(16), T.int64(0), v1 % T.int64(8) * T.int64(4) + v2 % T.int64(8) // T.int64(2), v2 % T.int64(16) // T.int64(8) * T.int64(4) + v1 % T.int64(16) // T.int64(8) * T.int64(2) + v2 % T.int64(2)])
                                    var_T_transpose_intermediate_reindex_shared_dyn_warp[v1 // T.int64(128), v2 // T.int64(32), v1 % T.int64(128) // T.int64(64), v2 % T.int64(32) // T.int64(16), v1 % T.int64(64) // T.int64(16), T.int64(0), v1 % T.int64(8) * T.int64(4) + v2 % T.int64(8) // T.int64(2), v2 % T.int64(16) // T.int64(8) * T.int64(4) + v1 % T.int64(16) // T.int64(8) * T.int64(2) + v2 % T.int64(2)] = var_T_transpose_intermediate_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(32), v1 % T.int64(128), v2 % T.int64(32)]
                            for ax1_0_3, ax2_0_3 in T.grid(T.int64(4), T.int64(4)):
                                with T.block("matmul_o_update"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(192), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(86) * T.int64(8) + ax1_0_2 * T.int64(4) + ax1_0_3)
                                    v2_o = T.axis.spatial(T.int64(688), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(86) * T.int64(8) + ax2_0_2 * T.int64(4) + ax2_0_3)
                                    v3_o = T.axis.reduce(T.int64(256), ax3_0_0 * T.int64(2) + ax3_0_1)
                                    T.reads(p_output0_intermediate_reindex_shared_dyn_warp[v1_o // T.int64(8), v2_o // T.int64(8), v1_o // T.int64(4) - v1_o // T.int64(8) * T.int64(2), v2_o // T.int64(4) - v2_o // T.int64(8) * T.int64(2), v1_o % T.int64(4), v2_o % T.int64(4), T.int64(0):T.int64(32), T.int64(0):T.int64(8)], A_reindex_shared_dyn_warp[v1_o * T.int64(16) // T.int64(128), v3_o * T.int64(16) // T.int64(32), (v1_o * T.int64(16) - T.int64(128) * (v1_o * T.int64(16) // T.int64(128))) // T.int64(64), (v3_o * T.int64(16) - T.int64(32) * (v3_o * T.int64(16) // T.int64(32))) // T.int64(16), (v1_o * T.int64(16) - T.int64(64) * (v1_o * T.int64(16) // T.int64(64))) // T.int64(16), T.int64(0), T.int64(0):T.int64(32), T.int64(0):T.int64(8)], var_T_transpose_intermediate_reindex_shared_dyn_warp[v2_o * T.int64(16) // T.int64(128), v3_o * T.int64(16) // T.int64(32), (v2_o * T.int64(16) - T.int64(128) * (v2_o * T.int64(16) // T.int64(128))) // T.int64(64), (v3_o * T.int64(16) - T.int64(32) * (v3_o * T.int64(16) // T.int64(32))) // T.int64(16), (v2_o * T.int64(16) - T.int64(64) * (v2_o * T.int64(16) // T.int64(64))) // T.int64(16), T.int64(0), T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.writes(p_output0_intermediate_reindex_shared_dyn_warp[v1_o // T.int64(8), v2_o // T.int64(8), v1_o // T.int64(4) - v1_o // T.int64(8) * T.int64(2), v2_o // T.int64(4) - v2_o // T.int64(8) * T.int64(2), v1_o % T.int64(4), v2_o % T.int64(4), T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    for ax1_1, ax2_1, ax3_1 in T.grid(T.int64(16), T.int64(16), T.int64(16)):
                                        with T.block("matmul"):
                                            v1_i, v2_i, v3_i = T.axis.remap("SSR", [ax1_1, ax2_1, ax3_1])
                                            T.reads(p_output0_intermediate_reindex_shared_dyn_warp[(v1_o * T.int64(16) + v1_i) // T.int64(128), (v2_o * T.int64(16) + v2_i) // T.int64(128), (v1_o * T.int64(16) + v1_i) % T.int64(128) // T.int64(64), (v2_o * T.int64(16) + v2_i) % T.int64(128) // T.int64(64), (v1_o * T.int64(16) + v1_i) % T.int64(64) // T.int64(16), (v2_o * T.int64(16) + v2_i) % T.int64(64) // T.int64(16), v1_i % T.int64(8) * T.int64(4) + v2_i % T.int64(8) // T.int64(2), v2_i % T.int64(16) // T.int64(8) * T.int64(4) + v1_i % T.int64(16) // T.int64(8) * T.int64(2) + v2_i % T.int64(2)], A_reindex_shared_dyn_warp[(v1_o * T.int64(16) + v1_i) // T.int64(128), (v3_o * T.int64(16) + v3_i) // T.int64(32), (v1_o * T.int64(16) + v1_i) % T.int64(128) // T.int64(64), (v3_o * T.int64(16) + v3_i) % T.int64(32) // T.int64(16), (v1_o * T.int64(16) + v1_i) % T.int64(64) // T.int64(16), T.int64(0), v1_i % T.int64(8) * T.int64(4) + v3_i % T.int64(8) // T.int64(2), v3_i % T.int64(16) // T.int64(8) * T.int64(4) + v1_i % T.int64(16) // T.int64(8) * T.int64(2) + v3_i % T.int64(2)], var_T_transpose_intermediate_reindex_shared_dyn_warp[(v2_o * T.int64(16) + v2_i) // T.int64(128), (v3_o * T.int64(16) + v3_i) // T.int64(32), (v2_o * T.int64(16) + v2_i) % T.int64(128) // T.int64(64), (v3_o * T.int64(16) + v3_i) % T.int64(32) // T.int64(16), (v2_o * T.int64(16) + v2_i) % T.int64(64) // T.int64(16), T.int64(0), v2_i % T.int64(8) * T.int64(4) + v3_i % T.int64(8) // T.int64(2), v3_i % T.int64(16) // T.int64(8) * T.int64(4) + v2_i % T.int64(16) // T.int64(8) * T.int64(2) + v3_i % T.int64(2)])
                                            T.writes(p_output0_intermediate_reindex_shared_dyn_warp[(v1_o * T.int64(16) + v1_i) // T.int64(128), (v2_o * T.int64(16) + v2_i) // T.int64(128), (v1_o * T.int64(16) + v1_i) % T.int64(128) // T.int64(64), (v2_o * T.int64(16) + v2_i) % T.int64(128) // T.int64(64), (v1_o * T.int64(16) + v1_i) % T.int64(64) // T.int64(16), (v2_o * T.int64(16) + v2_i) % T.int64(64) // T.int64(16), v1_i % T.int64(8) * T.int64(4) + v2_i % T.int64(8) // T.int64(2), v2_i % T.int64(16) // T.int64(8) * T.int64(4) + v1_i % T.int64(16) // T.int64(8) * T.int64(2) + v2_i % T.int64(2)])
                                            p_output0_intermediate_reindex_shared_dyn_warp[(v1_o * T.int64(16) + v1_i) // T.int64(128), (v2_o * T.int64(16) + v2_i) // T.int64(128), (v1_o * T.int64(16) + v1_i) % T.int64(128) // T.int64(64), (v2_o * T.int64(16) + v2_i) % T.int64(128) // T.int64(64), (v1_o * T.int64(16) + v1_i) % T.int64(64) // T.int64(16), (v2_o * T.int64(16) + v2_i) % T.int64(64) // T.int64(16), v1_i % T.int64(8) * T.int64(4) + v2_i % T.int64(8) // T.int64(2), v2_i % T.int64(16) // T.int64(8) * T.int64(4) + v1_i % T.int64(16) // T.int64(8) * T.int64(2) + v2_i % T.int64(2)] = p_output0_intermediate_reindex_shared_dyn_warp[(v1_o * T.int64(16) + v1_i) // T.int64(128), (v2_o * T.int64(16) + v2_i) // T.int64(128), (v1_o * T.int64(16) + v1_i) % T.int64(128) // T.int64(64), (v2_o * T.int64(16) + v2_i) % T.int64(128) // T.int64(64), (v1_o * T.int64(16) + v1_i) % T.int64(64) // T.int64(16), (v2_o * T.int64(16) + v2_i) % T.int64(64) // T.int64(16), v1_i % T.int64(8) * T.int64(4) + v2_i % T.int64(8) // T.int64(2), v2_i % T.int64(16) // T.int64(8) * T.int64(4) + v1_i % T.int64(16) // T.int64(8) * T.int64(2) + v2_i % T.int64(2)] + T.Cast("float32", A_reindex_shared_dyn_warp[(v1_o * T.int64(16) + v1_i) // T.int64(128), (v3_o * T.int64(16) + v3_i) // T.int64(32), (v1_o * T.int64(16) + v1_i) % T.int64(128) // T.int64(64), (v3_o * T.int64(16) + v3_i) % T.int64(32) // T.int64(16), (v1_o * T.int64(16) + v1_i) % T.int64(64) // T.int64(16), T.int64(0), v1_i % T.int64(8) * T.int64(4) + v3_i % T.int64(8) // T.int64(2), v3_i % T.int64(16) // T.int64(8) * T.int64(4) + v1_i % T.int64(16) // T.int64(8) * T.int64(2) + v3_i % T.int64(2)]) * T.Cast("float32", var_T_transpose_intermediate_reindex_shared_dyn_warp[(v2_o * T.int64(16) + v2_i) // T.int64(128), (v3_o * T.int64(16) + v3_i) // T.int64(32), (v2_o * T.int64(16) + v2_i) % T.int64(128) // T.int64(64), (v3_o * T.int64(16) + v3_i) % T.int64(32) // T.int64(16), (v2_o * T.int64(16) + v2_i) % T.int64(64) // T.int64(16), T.int64(0), v2_i % T.int64(8) * T.int64(4) + v3_i % T.int64(8) // T.int64(2), v3_i % T.int64(16) // T.int64(8) * T.int64(4) + v2_i % T.int64(16) // T.int64(8) * T.int64(2) + v3_i % T.int64(2)])
            for ax0 in range(T.int64(1)):
                for ax1_0 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax2_0 in range(T.int64(2)):
                        for ax1_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                            for ax2_1, ax1_2, ax2_2 in T.grid(T.int64(4), T.int64(16), T.int64(16)):
                                with T.block("p_output0_intermediate_reindex_shared.dyn_warp"):
                                    v0 = T.axis.spatial(T.int64(1), ax0)
                                    v1 = T.axis.spatial(T.int64(3072), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(86) * T.int64(128) + ax1_0 * T.int64(64) + ax1_1 * T.int64(16) + ax1_2)
                                    v2 = T.axis.spatial(T.int64(11008), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(86) * T.int64(128) + ax2_0 * T.int64(64) + ax2_1 * T.int64(16) + ax2_2)
                                    T.reads(p_output0_intermediate_reindex_shared_dyn_warp[v1 // T.int64(128), v2 // T.int64(128), v1 % T.int64(128) // T.int64(64), v2 % T.int64(128) // T.int64(64), v1 % T.int64(64) // T.int64(16), v2 % T.int64(64) // T.int64(16), v1 % T.int64(8) * T.int64(4) + v2 % T.int64(8) // T.int64(2), v2 % T.int64(16) // T.int64(8) * T.int64(4) + v1 % T.int64(16) // T.int64(8) * T.int64(2) + v2 % T.int64(2)])
                                    T.writes(p_output0_intermediate_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(128), v1 % T.int64(128), v2 % T.int64(128)])
                                    p_output0_intermediate_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(128), v1 % T.int64(128), v2 % T.int64(128)] = p_output0_intermediate_reindex_shared_dyn_warp[v1 // T.int64(128), v2 // T.int64(128), v1 % T.int64(128) // T.int64(64), v2 % T.int64(128) // T.int64(64), v1 % T.int64(64) // T.int64(16), v2 % T.int64(64) // T.int64(16), v1 % T.int64(8) * T.int64(4) + v2 % T.int64(8) // T.int64(2), v2 % T.int64(16) // T.int64(8) * T.int64(4) + v1 % T.int64(16) // T.int64(8) * T.int64(2) + v2 % T.int64(2)]
            for ax0_ax1_fused_0 in range(T.int64(16)):
                for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                            for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                with T.block("p_output0_intermediate_reindex_shared.dyn"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(T.int64(3072), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(86) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(128))
                                    v2 = T.axis.spatial(T.int64(11008), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(86) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(128))
                                    T.reads(p_output0_intermediate_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(128), v1 % T.int64(128), v2 % T.int64(128)])
                                    T.writes(var_compute_intermediate[v1 // T.int64(512), v1 % T.int64(512), v2])
                                    var_compute_intermediate[v1 // T.int64(512), v1 % T.int64(512), v2] = T.Cast("float16", p_output0_intermediate_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(128), v1 % T.int64(128), v2 % T.int64(128)])

    @R.function
    def main(A: R.Tensor((6, 512, 4096), dtype="float16"), B: R.Tensor((11008, 4096), dtype="float16")) -> R.Tensor((6, 512, 11008), dtype="float16"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.fused_fused_relax_permute_dims_relax_matmul_cast, (B, A), out_sinfo=R.Tensor((6, 512, 11008), dtype="float16"))
            R.output(gv)
        return gv
