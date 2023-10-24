# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_fused_decode3_fused_NT_matmul21_cast21_add5_silu2(lv2608: T.Buffer((T.int64(11008), T.int64(512)), "uint32"), lv2609: T.Buffer((T.int64(11008), T.int64(128)), "float16"), p_lv7330: T.handle, p_output0: T.handle, p_output0_intermediate_1: T.Buffer((T.int64(11008), T.int64(4096)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        b = T.int64()
        lv7330 = T.match_buffer(p_lv7330, (b, T.int64(512), T.int64(4096)), "float16")
        p_output0_intermediate = T.match_buffer(p_output0, (b, T.int64(512), T.int64(11008)), "float16")
        # with T.block("root"):
        lv7330_reindex_shared_dyn = T.alloc_buffer((b, T.int64(512), T.int64(4096)), "float16", scope="shared.dyn")
        lv7330_reindex_shared_dyn_warp = T.alloc_buffer((b, T.int64(32), T.int64(256), T.int64(32), T.int64(8)), "float16", scope="warp")
        p_output0_intermediate_1_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(11008), T.int64(4096)), "float16", scope="shared.dyn")
        p_output0_intermediate_1_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), T.int64(688), T.int64(256), T.int64(32), T.int64(8)), "float16", scope="warp")
        var_NT_matmul_intermediate_reindex_shared_dyn = T.alloc_buffer((b, T.int64(512), T.int64(11008)), scope="shared.dyn")
        var_NT_matmul_intermediate_reindex_shared_dyn_warp = T.alloc_buffer((b, T.int64(32), T.int64(688), T.int64(32), T.int64(8)), scope="warp")
        for i_j_fused_0 in T.thread_binding(T.int64(44032), thread="blockIdx.x"):
            for i_j_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for i_j_fused_2 in T.unroll(T.int64(2)):
                    for i_j_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("decode"):
                            v_i = T.axis.spatial(T.int64(11008), (i_j_fused_0 * T.int64(1024) + i_j_fused_1 * T.int64(8) + i_j_fused_2 * T.int64(4) + i_j_fused_3) // T.int64(4096))
                            v_j = T.axis.spatial(T.int64(4096), (i_j_fused_0 * T.int64(1024) + i_j_fused_1 * T.int64(8) + i_j_fused_2 * T.int64(4) + i_j_fused_3) % T.int64(4096))
                            T.reads(lv2608[v_i, v_j // T.int64(8)], lv2609[v_i, v_j // T.int64(32)])
                            T.writes(p_output0_intermediate_1[v_i, v_j])
                            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv2608[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv2609[v_i, v_j // T.int64(32)]
        for ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused in T.thread_binding(b * T.int64(344), thread="blockIdx.x"):
            for ax1_0_2_init in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                for ax2_0_2_init in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                    for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(4), T.int64(4)):
                        with T.block("NT_matmul_o_init"):
                            v0_o = T.axis.spatial(b, ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(344))
                            v1_o = T.axis.spatial(T.int64(32), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(344) // T.int64(86) * T.int64(8) + ax1_0_2_init * T.int64(4) + ax1_0_3_init)
                            v2_o = T.axis.spatial(T.int64(688), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(86) * T.int64(8) + ax2_0_2_init * T.int64(4) + ax2_0_3_init)
                            T.reads()
                            T.writes(var_NT_matmul_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                            with T.block("NT_matmul_init_o"):
                                v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                C_warp = T.match_buffer(var_NT_matmul_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=1)
                                for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    T.mma_fill("float32", 8, C_warp.data, C_warp.elem_offset)
            for ax3_0_0 in range(T.int64(128)):
                for ax0_ax1_fused_0 in range(T.int64(4)):
                    for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                        for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                            for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                    with T.block("lv7330_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(b, ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(344))
                                        v1 = T.axis.spatial(T.int64(512), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(344) // T.int64(86) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(4096), ax3_0_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(32))
                                        T.reads(lv7330[v0, v1, v2])
                                        T.writes(lv7330_reindex_shared_dyn[v0, v1, v2])
                                        T.block_attr({"permuted_layout": "g2s_A"})
                                        lv7330_reindex_shared_dyn[v0, v1, v2] = lv7330[v0, v1, v2]
                for ax0_ax1_fused_0 in range(T.int64(4)):
                    for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                        for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                            for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                    with T.block("p_output0_intermediate_1_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(11008), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(86) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(4096), ax3_0_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(32))
                                        T.reads(p_output0_intermediate_1[v1, v2])
                                        T.writes(p_output0_intermediate_1_reindex_shared_dyn[v0, v1, v2])
                                        T.block_attr({"permuted_layout": "g2s_B"})
                                        p_output0_intermediate_1_reindex_shared_dyn[v0, v1, v2] = p_output0_intermediate_1[v1, v2]
                for ax1_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax2_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax3_0_1 in range(T.int64(2)):
                            for ax0_0 in range(T.int64(4)):
                                with T.block("lv7330_reindex_shared.dyn_warp_o"):
                                    v0_o = T.axis.spatial(b, ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(344))
                                    v1_o = T.axis.spatial(T.int64(32), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(344) // T.int64(86) * T.int64(8) + ax1_0_2 * T.int64(4) + ax0_0)
                                    v2_o = T.axis.spatial(T.int64(256), ax3_0_0 * T.int64(2) + ax3_0_1)
                                    T.reads(lv7330_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    T.writes(lv7330_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.block_attr({"permuted_layout": "s2l_A"})
                                    with T.block("lv7330_reindex_shared.dyn_warp_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        T.reads(lv7330_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        T.writes(lv7330_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        warp = T.match_buffer(lv7330_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        shared = T.match_buffer(lv7330_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                        for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + T.int64(8) * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * T.int64(16), 1), shared.strides[0] * (tx % T.int64(16)) + T.int64(8) * (tx // T.int64(16)))
                            for ax0_0 in range(T.int64(4)):
                                with T.block("p_output0_intermediate_1_reindex_shared.dyn_warp_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(688), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(86) * T.int64(8) + ax2_0_2 * T.int64(4) + ax0_0)
                                    v2_o = T.axis.spatial(T.int64(256), ax3_0_0 * T.int64(2) + ax3_0_1)
                                    T.reads(p_output0_intermediate_1_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    T.writes(p_output0_intermediate_1_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.block_attr({"permuted_layout": "s2l_B"})
                                    with T.block("p_output0_intermediate_1_reindex_shared.dyn_warp_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        T.reads(p_output0_intermediate_1_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        T.writes(p_output0_intermediate_1_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        warp = T.match_buffer(p_output0_intermediate_1_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        shared = T.match_buffer(p_output0_intermediate_1_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                        for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + T.int64(8) * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * T.int64(16), 1), shared.strides[0] * T.int64(8) * (tx // T.int64(16)) + shared.strides[0] * (tx % T.int64(8)) + T.int64(8) * (tx % T.int64(16) // T.int64(8)))
                            for ax1_0_3, ax2_0_3 in T.grid(T.int64(4), T.int64(4)):
                                with T.block("NT_matmul_o_update"):
                                    v0_o = T.axis.spatial(b, ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(344))
                                    v1_o = T.axis.spatial(T.int64(32), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(344) // T.int64(86) * T.int64(8) + ax1_0_2 * T.int64(4) + ax1_0_3)
                                    v2_o = T.axis.spatial(T.int64(688), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(86) * T.int64(8) + ax2_0_2 * T.int64(4) + ax2_0_3)
                                    v3_o = T.axis.reduce(T.int64(256), ax3_0_0 * T.int64(2) + ax3_0_1)
                                    T.reads(var_NT_matmul_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], lv7330_reindex_shared_dyn_warp[v0_o, v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], p_output0_intermediate_1_reindex_shared_dyn_warp[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.writes(var_NT_matmul_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    with T.block("NT_matmul_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                        T.reads(var_NT_matmul_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], lv7330_reindex_shared_dyn_warp[v0_o, v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], p_output0_intermediate_1_reindex_shared_dyn_warp[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        T.writes(var_NT_matmul_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        A = T.match_buffer(lv7330_reindex_shared_dyn_warp[v0_o, v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        B = T.match_buffer(p_output0_intermediate_1_reindex_shared_dyn_warp[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        C = T.match_buffer(var_NT_matmul_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=8)
                                        for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A.data, A.elem_offset + tx * T.int64(8), B.data, B.elem_offset + tx * T.int64(8), C.data, C.elem_offset + tx * T.int64(8), T.bool(False))
                                            T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A.data, A.elem_offset + tx * T.int64(8), B.data, B.elem_offset + tx * T.int64(8) + T.int64(4), C.data, C.elem_offset + tx * T.int64(8) + T.int64(4), T.bool(False))
            for ax0 in range(T.int64(1)):
                for ax1_0 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax2_0 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax1_1, ax2_1 in T.grid(T.int64(4), T.int64(4)):
                            with T.block("var_NT_matmul_intermediate_reindex_shared.dyn_warp_o"):
                                v0_o = T.axis.spatial(b, ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(344) + ax0)
                                v1_o = T.axis.spatial(T.int64(32), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(344) // T.int64(86) * T.int64(8) + ax1_0 * T.int64(4) + ax1_1)
                                v2_o = T.axis.spatial(T.int64(688), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(86) * T.int64(8) + ax2_0 * T.int64(4) + ax2_1)
                                T.reads(var_NT_matmul_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                T.writes(var_NT_matmul_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                T.block_attr({"permuted_layout": "l2s_C"})
                                with T.block("var_NT_matmul_intermediate_reindex_shared.dyn_warp_o"):
                                    v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads(var_NT_matmul_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.writes(var_NT_matmul_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    C_warp = T.match_buffer(var_NT_matmul_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=1)
                                    C = T.match_buffer(var_NT_matmul_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=1)
                                    for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for local_id in range(T.int64(8)):
                                            C[T.int64(8) * (local_id % T.int64(4) // T.int64(2)) + tx // T.int64(4), T.int64(8) * (local_id // T.int64(4)) + tx % T.int64(4) * T.int64(2) + local_id % T.int64(2)] = C_warp[tx, local_id]
            for ax0_ax1_fused_0 in range(T.int64(16)):
                for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                            for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                with T.block("var_NT_matmul_intermediate_reindex_shared.dyn"):
                                    v0 = T.axis.spatial(b, ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(344))
                                    v1 = T.axis.spatial(T.int64(512), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(344) // T.int64(86) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(128))
                                    v2 = T.axis.spatial(T.int64(11008), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(86) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(128))
                                    T.reads(var_NT_matmul_intermediate_reindex_shared_dyn[v0, v1, v2])
                                    T.writes(p_output0_intermediate[v0, v1, v2])
                                    T.block_attr({"permuted_layout": "s2g_C"})
                                    p_output0_intermediate[v0, v1, v2] = T.Cast("float16", var_NT_matmul_intermediate_reindex_shared_dyn[v0, v1, v2])

    @R.function
    def main(w1: R.Tensor((11008, 512), dtype="uint32"), w2: R.Tensor((11008, 128), dtype="float16"), x: R.Tensor(("b", 512, 4096), dtype="float16")) -> R.Tensor(("b", 512, 11008), dtype="float16"):
        b = T.int64()
        cls = Module
        gv = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul21_cast21_add5_silu2, (w1, w2, x), out_sinfo=[R.Tensor((b, 512, 11008), dtype="float16"), R.Tensor((11008, 4096), dtype="float16")])
        out: R.Tensor((b, 512, 11008), dtype="float16") = gv[0]
        return out
