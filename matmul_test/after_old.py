# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_fused_relax_permute_dims_relax_matmul_cast(B: T.Buffer((T.int64(11008), T.int64(4096)), "float16"), A: T.Buffer((T.int64(6), T.int64(512), T.int64(4096)), "float16"), var_compute_intermediate: T.Buffer((T.int64(6), T.int64(512), T.int64(11008)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(3072), T.int64(4096)), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(192), T.int64(256), T.int64(32), T.int64(8)), "float16", scope="warp")
        var_T_transpose_intermediate_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(11008), T.int64(4096)), "float16", scope="shared.dyn")
        var_T_transpose_intermediate_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(688), T.int64(256), T.int64(32), T.int64(8)), "float16", scope="warp")
        p_output0_intermediate_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(3072), T.int64(11008)), scope="shared.dyn")
        p_output0_intermediate_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(192), T.int64(688), T.int64(32), T.int64(8)), scope="warp")
        for ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused in T.thread_binding(T.int64(2064), thread="blockIdx.x"):
            for ax1_0_2_init in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                for ax2_0_2_init in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                    for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(4), T.int64(4)):
                        with T.block("matmul_o_init"):
                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                            v1_o = T.axis.spatial(T.int64(192), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(86) * T.int64(8) + ax1_0_2_init * T.int64(4) + ax1_0_3_init)
                            v2_o = T.axis.spatial(T.int64(688), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(86) * T.int64(8) + ax2_0_2_init * T.int64(4) + ax2_0_3_init)
                            T.reads()
                            T.writes(p_output0_intermediate_reindex_shared_dyn_warp[v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                            with T.block("matmul_init_o"):
                                v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads()
                                T.writes(p_output0_intermediate_reindex_shared_dyn_warp[v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                C_warp = T.match_buffer(p_output0_intermediate_reindex_shared_dyn_warp[v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=1)
                                tx = T.launch_thread("threadIdx.x", 32)
                                T.mma_fill("float32", 8, C_warp.data, C_warp.elem_offset)
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
                                        T.writes(A_reindex_shared_dyn[v0, v1, v2])
                                        A_reindex_shared_dyn[v0, v1, v2] = A[v1 // T.int64(512), v1 % T.int64(512), v2]
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
                                        T.writes(var_T_transpose_intermediate_reindex_shared_dyn[v0, v1, v2])
                                        var_T_transpose_intermediate_reindex_shared_dyn[v0, v1, v2] = B[v1, v2]
                for ax1_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax2_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax3_0_1 in range(T.int64(2)):
                            for ax0_0 in range(T.int64(4)):
                                with T.block("A_reindex_shared.dyn_warp_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(192), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(86) * T.int64(8) + ax1_0_2 * T.int64(4) + ax0_0)
                                    v2_o = T.axis.spatial(T.int64(256), ax3_0_0 * T.int64(2) + ax3_0_1)
                                    T.reads(A_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    T.writes(A_reindex_shared_dyn_warp[v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    warp = T.match_buffer(A_reindex_shared_dyn_warp[v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                    shared = T.match_buffer(A_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                    tx = T.launch_thread("threadIdx.x", 32)
                                    T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + T.Cast("int64", 8 * tx), T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * T.int64(16), 1), shared.strides[0] * T.Cast("int64", tx % 16) + T.Cast("int64", 8 * (tx // 16)))
                            for ax0_0 in range(T.int64(4)):
                                with T.block("var_T_transpose_intermediate_reindex_shared.dyn_warp_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(688), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(86) * T.int64(8) + ax2_0_2 * T.int64(4) + ax0_0)
                                    v2_o = T.axis.spatial(T.int64(256), ax3_0_0 * T.int64(2) + ax3_0_1)
                                    T.reads(var_T_transpose_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    T.writes(var_T_transpose_intermediate_reindex_shared_dyn_warp[v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    warp = T.match_buffer(var_T_transpose_intermediate_reindex_shared_dyn_warp[v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                    shared = T.match_buffer(var_T_transpose_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                    tx = T.launch_thread("threadIdx.x", 32)
                                    T.ptx_ldmatrix("float16", T.bool(True), 4, ".b16", warp.data, warp.elem_offset + T.Cast("int64", 8 * tx), T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * T.int64(16), 1), shared.strides[0] * T.Cast("int64", tx % 16) + T.Cast("int64", 8 * (tx // 16)))
                            for ax1_0_3, ax2_0_3 in T.grid(T.int64(4), T.int64(4)):
                                with T.block("matmul_o_update"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(192), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(86) * T.int64(8) + ax1_0_2 * T.int64(4) + ax1_0_3)
                                    v2_o = T.axis.spatial(T.int64(688), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(86) * T.int64(8) + ax2_0_2 * T.int64(4) + ax2_0_3)
                                    v3_o = T.axis.reduce(T.int64(256), ax3_0_0 * T.int64(2) + ax3_0_1)
                                    T.reads(p_output0_intermediate_reindex_shared_dyn_warp[v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], A_reindex_shared_dyn_warp[v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], var_T_transpose_intermediate_reindex_shared_dyn_warp[v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.writes(p_output0_intermediate_reindex_shared_dyn_warp[v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    with T.block("matmul_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                        T.reads(p_output0_intermediate_reindex_shared_dyn_warp[v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], A_reindex_shared_dyn_warp[v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], var_T_transpose_intermediate_reindex_shared_dyn_warp[v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        T.writes(p_output0_intermediate_reindex_shared_dyn_warp[v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        A_1 = T.match_buffer(A_reindex_shared_dyn_warp[v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        B_1 = T.match_buffer(var_T_transpose_intermediate_reindex_shared_dyn_warp[v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        C = T.match_buffer(p_output0_intermediate_reindex_shared_dyn_warp[v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=8)
                                        tx = T.launch_thread("threadIdx.x", 32)
                                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_1.data, T.Cast("int64", A_1.elem_offset) + T.Cast("int64", tx) * T.int64(8), B_1.data, T.Cast("int64", B_1.elem_offset) + T.Cast("int64", tx) * T.int64(8), C.data, T.Cast("int64", C.elem_offset) + T.Cast("int64", tx) * T.int64(8), T.bool(False))
                                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_1.data, T.Cast("int64", A_1.elem_offset) + T.Cast("int64", tx) * T.int64(8), B_1.data, T.Cast("int64", B_1.elem_offset) + T.Cast("int64", tx) * T.int64(8) + T.int64(4), C.data, T.Cast("int64", C.elem_offset) + T.Cast("int64", tx) * T.int64(8) + T.int64(4), T.bool(False))
            for ax0 in range(T.int64(1)):
                for ax1_0 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax2_0 in range(T.int64(2)):
                        for ax1_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                            for ax2_1 in range(T.int64(4)):
                                with T.block("p_output0_intermediate_reindex_shared.dyn_warp_o"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0)
                                    v1_o = T.axis.spatial(T.int64(192), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(86) * T.int64(8) + ax1_0 * T.int64(4) + ax1_1)
                                    v2_o = T.axis.spatial(T.int64(688), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(86) * T.int64(8) + ax2_0 * T.int64(4) + ax2_1)
                                    T.reads(p_output0_intermediate_reindex_shared_dyn_warp[v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.writes(p_output0_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    C_warp = T.match_buffer(p_output0_intermediate_reindex_shared_dyn_warp[v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=1)
                                    C = T.match_buffer(p_output0_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=1)
                                    tx = T.launch_thread("threadIdx.x", 32)
                                    T.mma_store("float32", T.int64(16), T.int64(16), T.tvm_access_ptr(T.type_annotation("float32"), C.data, C.elem_offset, T.Cast("int64", C.strides[0]) * T.int64(16), 2), C_warp.data, C_warp.elem_offset, C.strides[0])
            for ax0_ax1_fused_0 in range(T.int64(16)):
                for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                            for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                with T.block("p_output0_intermediate_reindex_shared.dyn"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(T.int64(3072), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(86) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(128))
                                    v2 = T.axis.spatial(T.int64(11008), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(86) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(128))
                                    T.reads(p_output0_intermediate_reindex_shared_dyn[v0, v1, v2])
                                    T.writes(var_compute_intermediate[v1 // T.int64(512), v1 % T.int64(512), v2])
                                    var_compute_intermediate[v1 // T.int64(512), v1 % T.int64(512), v2] = T.Cast("float16", p_output0_intermediate_reindex_shared_dyn[v0, v1, v2])

    @R.function
    def main(A: R.Tensor((6, 512, 4096), dtype="float16"), B: R.Tensor((11008, 4096), dtype="float16")) -> R.Tensor((6, 512, 11008), dtype="float16"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.fused_fused_relax_permute_dims_relax_matmul_cast, (B, A), out_sinfo=R.Tensor((6, 512, 11008), dtype="float16"))
            R.output(gv)
        return gv
