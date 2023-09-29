# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_fused_relax_permute_dims_relax_matmul_cast(B: T.Buffer((T.int64(4096), T.int64(128)), "float16"), A: T.Buffer((T.int64(1), T.int64(512), T.int64(128)), "float16"), var_compute_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(4096)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_reindex_shared_dyn = T.alloc_buffer((T.int64(4), T.int64(4), T.int64(8), T.int64(2), T.int64(16), T.int64(16)), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(4), T.int64(4), T.int64(8), T.int64(2), T.int64(16), T.int64(16)), "float16", scope="wmma.matrix_a")
        var_T_transpose_intermediate_reindex_shared_dyn = T.alloc_buffer((T.int64(32), T.int64(4), T.int64(8), T.int64(2), T.int64(16), T.int64(16)), "float16", scope="shared.dyn")
        var_T_transpose_intermediate_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(32), T.int64(4), T.int64(8), T.int64(2), T.int64(16), T.int64(16)), "float16", scope="wmma.matrix_b")
        p_output0_intermediate_reindex_shared_dyn = T.alloc_buffer((T.int64(4), T.int64(32), T.int64(8), T.int64(8), T.int64(16), T.int64(16)), scope="shared.dyn")
        p_output0_intermediate_reindex_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(4), T.int64(32), T.int64(8), T.int64(8), T.int64(16), T.int64(16)), scope="wmma.accumulator")
        for ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused in T.thread_binding(T.int64(128), thread="blockIdx.x"):
            for ax1_0_2_init in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                for ax2_0_2_init in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                    for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(4), T.int64(4)):
                        with T.block("matmul_o_init"):
                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                            v1_o = T.axis.spatial(T.int64(32), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32) * T.int64(8) + ax1_0_2_init * T.int64(4) + ax1_0_3_init)
                            v2_o = T.axis.spatial(T.int64(256), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32) * T.int64(8) + ax2_0_2_init * T.int64(4) + ax2_0_3_init)
                            T.reads()
                            T.writes(p_output0_intermediate_reindex_shared_dyn_wmma_accumulator[v1_o // T.int64(8), v2_o // T.int64(8), v1_o % T.int64(8), v2_o % T.int64(8), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                            with T.block("matmul_init_o"):
                                v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads()
                                T.writes(p_output0_intermediate_reindex_shared_dyn_wmma_accumulator[v1_o // T.int64(8), v2_o // T.int64(8), v1_o % T.int64(8), v2_o % T.int64(8), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                C = T.match_buffer(p_output0_intermediate_reindex_shared_dyn_wmma_accumulator[v1_o // T.int64(8), v2_o // T.int64(8), v1_o % T.int64(8), v2_o % T.int64(8), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=8)
                                T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.float32(0))
            for ax3_0_0 in T.unroll(T.int64(4), annotations={"pragma_unroll_explicit": 0}):
                for ax0_ax1_fused_0 in range(T.int64(4)):
                    for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                        for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                            for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                    with T.block("A_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(512), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(128), ax3_0_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(32))
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(32), v1 % T.int64(128) // T.int64(16), v2 % T.int64(32) // T.int64(16), v1 % T.int64(16), v2 % T.int64(16)])
                                        T.block_attr({"buffer_dim_align": [[0, 4, 16, 8]]})
                                        A_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(32), v1 % T.int64(128) // T.int64(16), v2 % T.int64(32) // T.int64(16), v1 % T.int64(16), v2 % T.int64(16)] = A[v0, v1, v2]
                for ax0_ax1_fused_0 in range(T.int64(4)):
                    for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                        for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                            for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                    with T.block("var_T_transpose_intermediate_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(4096), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(128), ax3_0_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(32))
                                        T.reads(B[v1, v2])
                                        T.writes(var_T_transpose_intermediate_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(32), v1 % T.int64(128) // T.int64(16), v2 % T.int64(32) // T.int64(16), v1 % T.int64(16), v2 % T.int64(16)])
                                        T.block_attr({"buffer_dim_align": [[0, 4, 16, 8]]})
                                        var_T_transpose_intermediate_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(32), v1 % T.int64(128) // T.int64(16), v2 % T.int64(32) // T.int64(16), v1 % T.int64(16), v2 % T.int64(16)] = B[v1, v2]
                for ax1_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax2_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax3_0_1 in range(T.int64(2)):
                            for ax0 in range(T.int64(4)):
                                with T.block("A_reindex_shared.dyn_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(T.int64(4), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32))
                                    v1_o = T.axis.spatial(T.int64(4), ax3_0_0)
                                    v2_o = T.axis.spatial(T.int64(8), ax1_0_2 * T.int64(4) + ax0)
                                    v3_o = T.axis.spatial(T.int64(2), ax3_0_1)
                                    v4_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v5_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads(A_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                    T.writes(A_reindex_shared_dyn_wmma_matrix_a[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                    A_1 = T.match_buffer(A_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0 in range(T.int64(4)):
                                with T.block("var_T_transpose_intermediate_reindex_shared.dyn_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(T.int64(32), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32))
                                    v1_o = T.axis.spatial(T.int64(4), ax3_0_0)
                                    v2_o = T.axis.spatial(T.int64(8), ax2_0_2 * T.int64(4) + ax0)
                                    v3_o = T.axis.spatial(T.int64(2), ax3_0_1)
                                    v4_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v5_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads(var_T_transpose_intermediate_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                    T.writes(var_T_transpose_intermediate_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                    A_1 = T.match_buffer(var_T_transpose_intermediate_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C = T.match_buffer(var_T_transpose_intermediate_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "col_major")
                            for ax1_0_3, ax2_0_3 in T.grid(T.int64(4), T.int64(4)):
                                with T.block("matmul_o_update"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(32), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32) * T.int64(8) + ax1_0_2 * T.int64(4) + ax1_0_3)
                                    v2_o = T.axis.spatial(T.int64(256), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32) * T.int64(8) + ax2_0_2 * T.int64(4) + ax2_0_3)
                                    v3_o = T.axis.reduce(T.int64(8), ax3_0_0 * T.int64(2) + ax3_0_1)
                                    T.reads(p_output0_intermediate_reindex_shared_dyn_wmma_accumulator[v1_o // T.int64(8), v2_o // T.int64(8), v1_o % T.int64(8), v2_o % T.int64(8), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], A_reindex_shared_dyn_wmma_matrix_a[v1_o * T.int64(16) // T.int64(128), v3_o * T.int64(16) // T.int64(32), (v1_o * T.int64(16) - T.int64(128) * (v1_o * T.int64(16) // T.int64(128))) // T.int64(16), (v3_o * T.int64(16) - T.int64(32) * (v3_o * T.int64(16) // T.int64(32))) // T.int64(16), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], var_T_transpose_intermediate_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16) // T.int64(128), v3_o * T.int64(16) // T.int64(32), (v2_o * T.int64(16) - T.int64(128) * (v2_o * T.int64(16) // T.int64(128))) // T.int64(16), (v3_o * T.int64(16) - T.int64(32) * (v3_o * T.int64(16) // T.int64(32))) // T.int64(16), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                    T.writes(p_output0_intermediate_reindex_shared_dyn_wmma_accumulator[v1_o // T.int64(8), v2_o // T.int64(8), v1_o % T.int64(8), v2_o % T.int64(8), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                    with T.block("matmul_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                        T.reads(p_output0_intermediate_reindex_shared_dyn_wmma_accumulator[v1_o // T.int64(8), v2_o // T.int64(8), v1_o % T.int64(8), v2_o % T.int64(8), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], A_reindex_shared_dyn_wmma_matrix_a[v1_o // T.int64(8), v3_o // T.int64(2), v1_o % T.int64(8), v3_o % T.int64(2), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], var_T_transpose_intermediate_reindex_shared_dyn_wmma_matrix_b[v2_o // T.int64(8), v3_o // T.int64(2), v2_o % T.int64(8), v3_o % T.int64(2), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                        T.writes(p_output0_intermediate_reindex_shared_dyn_wmma_accumulator[v1_o // T.int64(8), v2_o // T.int64(8), v1_o % T.int64(8), v2_o % T.int64(8), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                        A_1 = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[v1_o // T.int64(8), v3_o // T.int64(2), v1_o % T.int64(8), v3_o % T.int64(2), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                        B_1 = T.match_buffer(var_T_transpose_intermediate_reindex_shared_dyn_wmma_matrix_b[v2_o // T.int64(8), v3_o // T.int64(2), v2_o % T.int64(8), v3_o % T.int64(2), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                        C = T.match_buffer(p_output0_intermediate_reindex_shared_dyn_wmma_accumulator[v1_o // T.int64(8), v2_o // T.int64(8), v1_o % T.int64(8), v2_o % T.int64(8), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=8)
                                        T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), A_1.data, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), B_1.data, B_1.elem_offset // B_1.strides[0] // T.int64(16) * (B_1.strides[0] // T.int64(16)) + B_1.elem_offset % B_1.strides[0] // T.int64(16), C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16))
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_0 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax3_0 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax2_1, ax3_1 in T.grid(T.int64(4), T.int64(4)):
                            with T.block("p_output0_intermediate_reindex_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(4), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32) + ax0)
                                v1_o = T.axis.spatial(T.int64(32), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32) + ax1)
                                v2_o = T.axis.spatial(T.int64(8), ax2_0 * T.int64(4) + ax2_1)
                                v3_o = T.axis.spatial(T.int64(8), ax3_0 * T.int64(4) + ax3_1)
                                v4_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v5_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads(p_output0_intermediate_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                T.writes(p_output0_intermediate_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                A_1 = T.match_buffer(p_output0_intermediate_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=8)
                                C = T.match_buffer(p_output0_intermediate_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=8)
                                T.tvm_store_matrix_sync(A_1.data, 16, 16, 16, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float32"), C.data, C.elem_offset, C.strides[0] * T.int64(16), 2), C.strides[0], "row_major")
            for ax0_ax1_fused_0 in range(T.int64(16)):
                for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                            for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                with T.block("p_output0_intermediate_reindex_shared.dyn"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(T.int64(512), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(128))
                                    v2 = T.axis.spatial(T.int64(4096), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(128))
                                    T.reads(p_output0_intermediate_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(128), v1 % T.int64(128) // T.int64(16), v2 % T.int64(128) // T.int64(16), v1 % T.int64(16), v2 % T.int64(16)])
                                    T.writes(var_compute_intermediate[T.int64(0), v1, v2])
                                    var_compute_intermediate[T.int64(0), v1, v2] = T.Cast("float16", p_output0_intermediate_reindex_shared_dyn[v1 // T.int64(128), v2 // T.int64(128), v1 % T.int64(128) // T.int64(16), v2 % T.int64(128) // T.int64(16), v1 % T.int64(16), v2 % T.int64(16)])

    @R.function
    def main(A: R.Tensor((1, 512, 128), dtype="float16"), B: R.Tensor((4096, 128), dtype="float16")) -> R.Tensor((1, 512, 4096), dtype="float16"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.fused_fused_relax_permute_dims_relax_matmul_cast, (B, A), out_sinfo=R.Tensor((1, 512, 4096), dtype="float16"))
            R.output(gv)
        return gv
