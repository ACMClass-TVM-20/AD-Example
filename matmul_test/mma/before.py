# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_matmul_cast(A: T.Buffer((T.int64(4), T.int64(511), T.int64(4096)), "float16"), B: T.Buffer((T.int64(4096), T.int64(4094)), "float16"), var_compute_intermediate: T.Buffer((T.int64(4), T.int64(511), T.int64(4094)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_matmul_intermediate = T.alloc_buffer((T.int64(4), T.int64(511), T.int64(4094)))
        for i0, i1, i2, k in T.grid(T.int64(4), T.int64(511), T.int64(4094), T.int64(4096)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B[v_k, v_i2])
                T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", A[v_i0, v_i1, v_k]) * T.Cast("float32", B[v_k, v_i2])
        for i0, i1, i2 in T.grid(T.int64(4), T.int64(511), T.int64(4094)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
                T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
                var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_matmul_intermediate[v_i0, v_i1, v_i2])

    @R.function
    def main(A: R.Tensor((4, 511, 4096), dtype="float16"), B: R.Tensor((4096, 4094), dtype="float16")) -> R.Tensor((4, 511, 4094), dtype="float16"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.fused_matmul_cast, (A, B), out_sinfo=R.Tensor((4, 511, 4094), dtype="float16"))
            R.output(gv)
        return gv
