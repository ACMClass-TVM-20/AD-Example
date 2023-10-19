# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_fused_dequantize_fused_relax_permute_dims_relax_matmul_cast(weight: T.Buffer((T.int64(4096), T.int64(512)), "uint32"), scale: T.Buffer((T.int64(4096), T.int64(128)), "float16"), A: T.Buffer((T.int64(1), T.int64(4094), T.int64(4096)), "float16"), var_compute_intermediate: T.Buffer((T.int64(1), T.int64(4094), T.int64(4096)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
        var_T_transpose_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
        p_output0_intermediate = T.alloc_buffer((T.int64(1), T.int64(4094), T.int64(4096)))
        for i, j in T.grid(T.int64(4096), T.int64(4096)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(weight[v_i, v_j // T.int64(8)], scale[v_i, v_j // T.int64(32)])
                T.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(weight[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * scale[v_i, v_j // T.int64(32)]
        for ax0, ax1 in T.grid(T.int64(4096), T.int64(4096)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(var_decode_intermediate[v_ax1, v_ax0])
                T.writes(var_T_transpose_intermediate[v_ax0, v_ax1])
                var_T_transpose_intermediate[v_ax0, v_ax1] = var_decode_intermediate[v_ax1, v_ax0]
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(4094), T.int64(4096), T.int64(4096)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], var_T_transpose_intermediate[v_k, v_i2])
                T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    p_output0_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                p_output0_intermediate[v_i0, v_i1, v_i2] = p_output0_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", A[v_i0, v_i1, v_k]) * T.Cast("float32", var_T_transpose_intermediate[v_k, v_i2])
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(4094), T.int64(4096)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(p_output0_intermediate[v_i0, v_i1, v_i2])
                T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
                var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", p_output0_intermediate[v_i0, v_i1, v_i2])

    @R.function
    def main(A: R.Tensor((1, 4094, 4096), dtype="float16"), weight: R.Tensor((4096, 512), dtype="uint32"), scale: R.Tensor((4096, 128), dtype="float16")) -> R.Tensor((1, 4094, 4096), dtype="float16"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.fused_fused_dequantize_fused_relax_permute_dims_relax_matmul_cast, (weight, scale, A), out_sinfo=R.Tensor((1, 4094, 4096), dtype="float16"))
            R.output(gv)
        return gv
