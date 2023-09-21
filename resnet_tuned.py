from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

@I.ir_module
class Module:
    I.module_attrs({"input_num": 1, "param_num": 62, "state_num": 40})
    @T.prim_func
    def add(A: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(64), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(9.9999997473787516e-06)

    @T.prim_func
    def add1(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), B: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(32), ax3)
                v_ax3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def add10(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), B: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(4), ax3)
                v_ax3 = T.axis.spatial(T.int64(4), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def add11(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), B: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), T_add: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(4), ax3)
                v_ax3 = T.axis.spatial(T.int64(4), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def add12(A: T.Buffer((T.int64(32), T.int64(10)), "float32"), B: T.Buffer((T.int64(10),), "float32"), T_add: T.Buffer((T.int64(32), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, vi in T.grid(T.int64(32), T.int64(10), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_ax0, v_ax1], B[v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[v_ax1]

    @T.prim_func
    def add13(A: T.Buffer((T.int64(64),), "float32"), B: T.Buffer((T.int64(64),), "float32"), T_add: T.Buffer((T.int64(64),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(64), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), T.int64(0))
                T.reads(A[v_ax0], B[v_ax0])
                T.writes(T_add[v_ax0])
                T_add[v_ax0] = A[v_ax0] + B[v_ax0]

    @T.prim_func
    def add14(A: T.Buffer((T.int64(128),), "float32"), B: T.Buffer((T.int64(128),), "float32"), T_add: T.Buffer((T.int64(128),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(128), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), T.int64(0))
                T.reads(A[v_ax0], B[v_ax0])
                T.writes(T_add[v_ax0])
                T_add[v_ax0] = A[v_ax0] + B[v_ax0]

    @T.prim_func
    def add15(A: T.Buffer((T.int64(256),), "float32"), B: T.Buffer((T.int64(256),), "float32"), T_add: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(256), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), T.int64(0))
                T.reads(A[v_ax0], B[v_ax0])
                T.writes(T_add[v_ax0])
                T_add[v_ax0] = A[v_ax0] + B[v_ax0]

    @T.prim_func
    def add16(A: T.Buffer((T.int64(512),), "float32"), B: T.Buffer((T.int64(512),), "float32"), T_add: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(512), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), T.int64(0))
                T.reads(A[v_ax0], B[v_ax0])
                T.writes(T_add[v_ax0])
                T_add[v_ax0] = A[v_ax0] + B[v_ax0]

    @T.prim_func
    def add17(A: T.Buffer((), "int64"), T_add: T.Buffer((), "int64")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for vi in range(T.int64(1)):
            with T.block("T_add"):
                vi_1 = T.axis.spatial(1, T.int64(0))
                T.reads(A[()])
                T.writes(T_add[()])
                T_add[()] = A[()] + T.int64(1)

    @T.prim_func
    def add18(A: T.Buffer((T.int64(64), T.int64(3), T.int64(3), T.int64(3)), "float32"), B: T.Buffer((T.int64(64), T.int64(3), T.int64(3), T.int64(3)), "float32"), T_add: T.Buffer((T.int64(64), T.int64(3), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(64), T.int64(3), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), ax1)
                v_ax1, v_ax2 = T.axis.remap("SS", [ax2, ax3])
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def add19(A: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32"), B: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32"), T_add: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(64), T.int64(64), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def add2(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), B: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), T_add: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(32), ax3)
                v_ax3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def add20(A: T.Buffer((T.int64(128), T.int64(64), T.int64(3), T.int64(3)), "float32"), B: T.Buffer((T.int64(128), T.int64(64), T.int64(3), T.int64(3)), "float32"), T_add: T.Buffer((T.int64(128), T.int64(64), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(128), T.int64(64), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def add21(A: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32"), B: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32"), T_add: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(128), T.int64(128), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def add22(A: T.Buffer((T.int64(128), T.int64(64), T.int64(1), T.int64(1)), "float32"), B: T.Buffer((T.int64(128), T.int64(64), T.int64(1), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(128), T.int64(64), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(128), T.int64(64), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def add23(A: T.Buffer((T.int64(256), T.int64(128), T.int64(3), T.int64(3)), "float32"), B: T.Buffer((T.int64(256), T.int64(128), T.int64(3), T.int64(3)), "float32"), T_add: T.Buffer((T.int64(256), T.int64(128), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(256), T.int64(128), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def add24(A: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), B: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), T_add: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(256), T.int64(256), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def add25(A: T.Buffer((T.int64(256), T.int64(128), T.int64(1), T.int64(1)), "float32"), B: T.Buffer((T.int64(256), T.int64(128), T.int64(1), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(256), T.int64(128), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(256), T.int64(128), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def add26(A: T.Buffer((T.int64(512), T.int64(256), T.int64(3), T.int64(3)), "float32"), B: T.Buffer((T.int64(512), T.int64(256), T.int64(3), T.int64(3)), "float32"), T_add: T.Buffer((T.int64(512), T.int64(256), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(512), T.int64(256), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def add27(A: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), B: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), T_add: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(512), T.int64(512), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def add28(A: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32"), B: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(512), T.int64(256), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def add29(A: T.Buffer((T.int64(512), T.int64(10)), "float32"), B: T.Buffer((T.int64(512), T.int64(10)), "float32"), T_add: T.Buffer((T.int64(512), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, vi in T.grid(T.int64(512), T.int64(10), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[v_ax0, v_ax1]

    @T.prim_func
    def add3(A: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(128), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(9.9999997473787516e-06)

    @T.prim_func
    def add30(A: T.Buffer((T.int64(10),), "float32"), B: T.Buffer((T.int64(10),), "float32"), T_add: T.Buffer((T.int64(10),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(10), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_ax0], B[v_ax0])
                T.writes(T_add[v_ax0])
                T_add[v_ax0] = A[v_ax0] + B[v_ax0]

    @T.prim_func
    def add4(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), B: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(16), ax3)
                v_ax3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def add5(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), B: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), T_add: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(16), ax3)
                v_ax3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def add6(A: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(256), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(9.9999997473787516e-06)

    @T.prim_func
    def add7(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), B: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(8), ax3)
                v_ax3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def add8(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), B: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), T_add: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(8), ax3)
                v_ax3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def add9(A: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(512), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_add"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(9.9999997473787516e-06)

    @T.prim_func
    def avg_pool2d(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), pool_avg: T.Buffer((T.int64(32), T.int64(512), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pool_sum = T.alloc_buffer((T.int64(32), T.int64(512), T.int64(1), T.int64(1)))
        for ax0, ax1, ax2, ax3, rv0, rv1, vi in T.grid(T.int64(32), T.int64(512), T.int64(1), T.int64(1), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("pool_sum"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), rv0)
                v_rv0 = T.axis.reduce(T.int64(4), rv1)
                v_rv1 = T.axis.reduce(T.int64(4), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2 + v_rv0, v_ax3 + v_rv1])
                T.writes(pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                with T.init():
                    pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] = pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] + A[v_ax0, v_ax1, v_ax2 + v_rv0, v_ax3 + v_rv1]
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(512), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("pool_avg"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(pool_avg[v_ax0, v_ax1, v_ax2, v_ax3])
                T.block_attr({"schedule_rule": "meta_schedule.pool_avg"})
                pool_avg[v_ax0, v_ax1, v_ax2, v_ax3] = pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] / T.Cast("float32", (T.min(T.int64(3), T.int64(3) - v_ax2) + T.int64(1)) * (T.min(T.int64(3), T.int64(3) - v_ax3) + T.int64(1)))

    @T.prim_func
    def avg_pool2d_backward(A: T.Buffer((T.int64(32), T.int64(512), T.int64(1), T.int64(1)), "float32"), B: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), T_pool_grad: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, wh, ww, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("T_pool_grad"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2, v_ax3, v_wh = T.axis.remap("SSR", [ax3, wh, ww])
                v_ww = T.axis.reduce(T.int64(4), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2 - v_wh, v_ax3 - v_ww])
                T.writes(T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3])
                with T.init():
                    T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] = T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] + T.if_then_else(T.Select(v_ax2 < T.int64(4), T.int64(0), v_ax2 - T.int64(3)) <= v_ax2 - v_wh and v_ax2 - v_wh < T.int64(1) and T.Select(v_ax3 < T.int64(4), T.int64(0), v_ax3 - T.int64(3)) <= v_ax3 - v_ww and v_ax3 - v_ww < T.int64(1), A[v_ax0, v_ax1, v_ax2 - v_wh, v_ax3 - v_ww] * T.float32(0.0625), T.float32(0))

    @T.prim_func
    def broadcast_to(A: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), T_broadcast_to: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("T_broadcast_to"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(4), ax3)
                v_ax3 = T.axis.spatial(T.int64(4), T.int64(0))
                T.reads(A[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2, v_ax3])
                T_broadcast_to[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def broadcast_to1(A: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), T_broadcast_to: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("T_broadcast_to"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(8), ax3)
                v_ax3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(A[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2, v_ax3])
                T_broadcast_to[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def broadcast_to2(A: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32"), T_broadcast_to: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("T_broadcast_to"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(16), ax3)
                v_ax3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(A[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2, v_ax3])
                T_broadcast_to[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def broadcast_to3(A: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32"), T_broadcast_to: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("T_broadcast_to"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(32), ax3)
                v_ax3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2, v_ax3])
                T_broadcast_to[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def collapse_sum(A: T.Buffer((T.int64(32), T.int64(10)), "float32"), A_red: T.Buffer((T.int64(10),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, k0, vi in T.grid(T.int64(10), T.int64(32), T.int64(1)):
            with T.block("A_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(10), k0)
                v_k0 = T.axis.reduce(T.int64(32), T.int64(0))
                T.reads(A[v_k0, v_ax0])
                T.writes(A_red[v_ax0])
                with T.init():
                    A_red[v_ax0] = T.float32(0)
                A_red[v_ax0] = A_red[v_ax0] + A[v_k0, v_ax0]

    @T.prim_func
    def collapse_sum1(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), A_red: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, k0, k2, k3, vi in T.grid(T.int64(1), T.int64(512), T.int64(1), T.int64(1), T.int64(32), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("A_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), k0)
                v_k0 = T.axis.reduce(T.int64(32), k2)
                v_k2 = T.axis.reduce(T.int64(4), k3)
                v_k3 = T.axis.reduce(T.int64(4), T.int64(0))
                T.reads(A[v_k0, v_ax1, v_k2, v_k3])
                T.writes(A_red[v_ax0, v_ax1, v_ax2, v_ax3])
                with T.init():
                    A_red[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                A_red[v_ax0, v_ax1, v_ax2, v_ax3] = A_red[v_ax0, v_ax1, v_ax2, v_ax3] + A[v_k0, v_ax1, v_k2, v_k3]

    @T.prim_func
    def collapse_sum2(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), A_red: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, k0, k2, k3, vi in T.grid(T.int64(1), T.int64(256), T.int64(1), T.int64(1), T.int64(32), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("A_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), k0)
                v_k0 = T.axis.reduce(T.int64(32), k2)
                v_k2 = T.axis.reduce(T.int64(8), k3)
                v_k3 = T.axis.reduce(T.int64(8), T.int64(0))
                T.reads(A[v_k0, v_ax1, v_k2, v_k3])
                T.writes(A_red[v_ax0, v_ax1, v_ax2, v_ax3])
                with T.init():
                    A_red[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                A_red[v_ax0, v_ax1, v_ax2, v_ax3] = A_red[v_ax0, v_ax1, v_ax2, v_ax3] + A[v_k0, v_ax1, v_k2, v_k3]

    @T.prim_func
    def collapse_sum3(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), A_red: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, k0, k2, k3, vi in T.grid(T.int64(1), T.int64(128), T.int64(1), T.int64(1), T.int64(32), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("A_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), k0)
                v_k0 = T.axis.reduce(T.int64(32), k2)
                v_k2 = T.axis.reduce(T.int64(16), k3)
                v_k3 = T.axis.reduce(T.int64(16), T.int64(0))
                T.reads(A[v_k0, v_ax1, v_k2, v_k3])
                T.writes(A_red[v_ax0, v_ax1, v_ax2, v_ax3])
                with T.init():
                    A_red[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                A_red[v_ax0, v_ax1, v_ax2, v_ax3] = A_red[v_ax0, v_ax1, v_ax2, v_ax3] + A[v_k0, v_ax1, v_k2, v_k3]

    @T.prim_func
    def collapse_sum4(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), A_red: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, k0, k2, k3, vi in T.grid(T.int64(1), T.int64(64), T.int64(1), T.int64(1), T.int64(32), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("A_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), k0)
                v_k0, v_k2 = T.axis.remap("RR", [k2, k3])
                v_k3 = T.axis.reduce(T.int64(32), T.int64(0))
                T.reads(A[v_k0, v_ax1, v_k2, v_k3])
                T.writes(A_red[v_ax0, v_ax1, v_ax2, v_ax3])
                with T.init():
                    A_red[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                A_red[v_ax0, v_ax1, v_ax2, v_ax3] = A_red[v_ax0, v_ax1, v_ax2, v_ax3] + A[v_k0, v_ax1, v_k2, v_k3]

    @T.prim_func
    def conv2d(A: T.Buffer((T.int64(32), T.int64(3), T.int64(32), T.int64(32)), "float32"), B: T.Buffer((T.int64(64), T.int64(3), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(3), T.int64(34), T.int64(34)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(3), T.int64(34), T.int64(34), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(3), i2)
                v_i2 = T.axis.spatial(T.int64(34), i3)
                v_i3 = T.axis.spatial(T.int64(34), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(33) and T.int64(1) <= v_i3 and v_i3 < T.int64(33), A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for nn, ff, yy, xx, rc, ry, rx, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(3), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("conv2d_nchw"):
                vi = T.axis.spatial(T.int64(1), nn)
                v_nn = T.axis.spatial(T.int64(32), ff)
                v_ff = T.axis.spatial(T.int64(64), yy)
                v_yy = T.axis.spatial(T.int64(32), xx)
                v_xx = T.axis.spatial(T.int64(32), rc)
                v_rc, v_ry = T.axis.remap("RR", [ry, rx])
                v_rx = T.axis.reduce(T.int64(3), T.int64(0))
                T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func
    def conv2d1(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), B: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(64), T.int64(34), T.int64(34)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(64), T.int64(34), T.int64(34), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(64), i2)
                v_i2 = T.axis.spatial(T.int64(34), i3)
                v_i3 = T.axis.spatial(T.int64(34), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(33) and T.int64(1) <= v_i3 and v_i3 < T.int64(33), A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for nn, ff, yy, xx, rc, ry, rx, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(64), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("conv2d_nchw"):
                vi = T.axis.spatial(T.int64(1), nn)
                v_nn = T.axis.spatial(T.int64(32), ff)
                v_ff = T.axis.spatial(T.int64(64), yy)
                v_yy = T.axis.spatial(T.int64(32), xx)
                v_xx = T.axis.spatial(T.int64(32), rc)
                v_rc = T.axis.reduce(T.int64(64), ry)
                v_ry = T.axis.reduce(T.int64(3), rx)
                v_rx = T.axis.reduce(T.int64(3), T.int64(0))
                T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func
    def conv2d10(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), B: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(256), i2)
                v_i2 = T.axis.spatial(T.int64(8), i3)
                v_i3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, v_i2, v_i3]
        for nn, ff, yy, xx, rc, ry, rx, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(256), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("conv2d_nchw"):
                vi = T.axis.spatial(T.int64(1), nn)
                v_nn = T.axis.spatial(T.int64(32), ff)
                v_ff = T.axis.spatial(T.int64(512), yy)
                v_yy = T.axis.spatial(T.int64(4), xx)
                v_xx = T.axis.spatial(T.int64(4), rc)
                v_rc = T.axis.reduce(T.int64(256), ry)
                v_ry = T.axis.reduce(T.int64(1), rx)
                v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func
    def conv2d11(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), B: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), conv2d_cnhw: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(512), T.int64(6), T.int64(6)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(512), T.int64(6), T.int64(6), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(512), i2)
                v_i2 = T.axis.spatial(T.int64(6), i3)
                v_i3 = T.axis.spatial(T.int64(6), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(5) and T.int64(1) <= v_i3 and v_i3 < T.int64(5), A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for ff, nn, yy, xx, rc, ry, rx, vi in T.grid(T.int64(512), T.int64(512), T.int64(3), T.int64(3), T.int64(32), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("conv2d_cnhw"):
                vi = T.axis.spatial(T.int64(1), ff)
                v_ff = T.axis.spatial(T.int64(512), nn)
                v_nn = T.axis.spatial(T.int64(512), yy)
                v_yy = T.axis.spatial(T.int64(3), xx)
                v_xx = T.axis.spatial(T.int64(3), rc)
                v_rc = T.axis.reduce(T.int64(32), ry)
                v_ry = T.axis.reduce(T.int64(4), rx)
                v_rx = T.axis.reduce(T.int64(4), T.int64(0))
                T.reads(pad_temp[v_rc, v_nn, v_yy + v_ry, v_xx + v_rx], B[v_rc, v_ff, v_ry, v_rx])
                T.writes(conv2d_cnhw[v_ff, v_nn, v_yy, v_xx])
                with T.init():
                    conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = T.float32(0)
                conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] + pad_temp[v_rc, v_nn, v_yy + v_ry, v_xx + v_rx] * B[v_rc, v_ff, v_ry, v_rx]

    @T.prim_func
    def conv2d12(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), B: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), conv2d_cnhw: T.Buffer((T.int64(512), T.int64(256), T.int64(2), T.int64(2)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(256), i2)
                v_i2 = T.axis.spatial(T.int64(8), i3)
                v_i3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, v_i2, v_i3]
        for ff, nn, yy, xx, rc, ry, rx, vi in T.grid(T.int64(512), T.int64(256), T.int64(2), T.int64(2), T.int64(32), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("conv2d_cnhw"):
                vi = T.axis.spatial(T.int64(1), ff)
                v_ff = T.axis.spatial(T.int64(512), nn)
                v_nn = T.axis.spatial(T.int64(256), yy)
                v_yy = T.axis.spatial(T.int64(2), xx)
                v_xx = T.axis.spatial(T.int64(2), rc)
                v_rc = T.axis.reduce(T.int64(32), ry)
                v_ry = T.axis.reduce(T.int64(4), rx)
                v_rx = T.axis.reduce(T.int64(4), T.int64(0))
                T.reads(pad_temp[v_rc, v_nn, v_ry * T.int64(2) + v_yy, v_rx * T.int64(2) + v_xx], B[v_rc, v_ff, v_ry, v_rx])
                T.writes(conv2d_cnhw[v_ff, v_nn, v_yy, v_xx])
                with T.init():
                    conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = T.float32(0)
                conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] + pad_temp[v_rc, v_nn, v_ry * T.int64(2) + v_yy, v_rx * T.int64(2) + v_xx] * B[v_rc, v_ff, v_ry, v_rx]

    @T.prim_func
    def conv2d13(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), B: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), conv2d_cnhw: T.Buffer((T.int64(512), T.int64(256), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(256), T.int64(10), T.int64(10)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(256), T.int64(10), T.int64(10), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(256), i2)
                v_i2 = T.axis.spatial(T.int64(10), i3)
                v_i3 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(9) and T.int64(1) <= v_i3 and v_i3 < T.int64(9), A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for ff, nn, yy, xx, rc, ry, rx, vi in T.grid(T.int64(512), T.int64(256), T.int64(4), T.int64(4), T.int64(32), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("conv2d_cnhw"):
                vi = T.axis.spatial(T.int64(1), ff)
                v_ff = T.axis.spatial(T.int64(512), nn)
                v_nn = T.axis.spatial(T.int64(256), yy)
                v_yy = T.axis.spatial(T.int64(4), xx)
                v_xx = T.axis.spatial(T.int64(4), rc)
                v_rc = T.axis.reduce(T.int64(32), ry)
                v_ry = T.axis.reduce(T.int64(4), rx)
                v_rx = T.axis.reduce(T.int64(4), T.int64(0))
                T.reads(pad_temp[v_rc, v_nn, v_ry * T.int64(2) + v_yy, v_rx * T.int64(2) + v_xx], B[v_rc, v_ff, v_ry, v_rx])
                T.writes(conv2d_cnhw[v_ff, v_nn, v_yy, v_xx])
                with T.init():
                    conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = T.float32(0)
                conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] + pad_temp[v_rc, v_nn, v_ry * T.int64(2) + v_yy, v_rx * T.int64(2) + v_xx] * B[v_rc, v_ff, v_ry, v_rx]

    @T.prim_func
    def conv2d14(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), B: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), conv2d_cnhw: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(256), T.int64(10), T.int64(10)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(256), T.int64(10), T.int64(10), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(256), i2)
                v_i2 = T.axis.spatial(T.int64(10), i3)
                v_i3 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(9) and T.int64(1) <= v_i3 and v_i3 < T.int64(9), A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for ff, nn, yy, xx, rc, ry, rx, vi in T.grid(T.int64(256), T.int64(256), T.int64(3), T.int64(3), T.int64(32), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("conv2d_cnhw"):
                vi = T.axis.spatial(T.int64(1), ff)
                v_ff = T.axis.spatial(T.int64(256), nn)
                v_nn = T.axis.spatial(T.int64(256), yy)
                v_yy = T.axis.spatial(T.int64(3), xx)
                v_xx = T.axis.spatial(T.int64(3), rc)
                v_rc = T.axis.reduce(T.int64(32), ry)
                v_ry = T.axis.reduce(T.int64(8), rx)
                v_rx = T.axis.reduce(T.int64(8), T.int64(0))
                T.reads(pad_temp[v_rc, v_nn, v_yy + v_ry, v_xx + v_rx], B[v_rc, v_ff, v_ry, v_rx])
                T.writes(conv2d_cnhw[v_ff, v_nn, v_yy, v_xx])
                with T.init():
                    conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = T.float32(0)
                conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] + pad_temp[v_rc, v_nn, v_yy + v_ry, v_xx + v_rx] * B[v_rc, v_ff, v_ry, v_rx]

    @T.prim_func
    def conv2d15(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), B: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), conv2d_cnhw: T.Buffer((T.int64(256), T.int64(128), T.int64(2), T.int64(2)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(128), i2)
                v_i2 = T.axis.spatial(T.int64(16), i3)
                v_i3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, v_i2, v_i3]
        for ff, nn, yy, xx, rc, ry, rx, vi in T.grid(T.int64(256), T.int64(128), T.int64(2), T.int64(2), T.int64(32), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("conv2d_cnhw"):
                vi = T.axis.spatial(T.int64(1), ff)
                v_ff = T.axis.spatial(T.int64(256), nn)
                v_nn = T.axis.spatial(T.int64(128), yy)
                v_yy = T.axis.spatial(T.int64(2), xx)
                v_xx = T.axis.spatial(T.int64(2), rc)
                v_rc = T.axis.reduce(T.int64(32), ry)
                v_ry = T.axis.reduce(T.int64(8), rx)
                v_rx = T.axis.reduce(T.int64(8), T.int64(0))
                T.reads(pad_temp[v_rc, v_nn, v_ry * T.int64(2) + v_yy, v_rx * T.int64(2) + v_xx], B[v_rc, v_ff, v_ry, v_rx])
                T.writes(conv2d_cnhw[v_ff, v_nn, v_yy, v_xx])
                with T.init():
                    conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = T.float32(0)
                conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] + pad_temp[v_rc, v_nn, v_ry * T.int64(2) + v_yy, v_rx * T.int64(2) + v_xx] * B[v_rc, v_ff, v_ry, v_rx]

    @T.prim_func
    def conv2d16(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), B: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), conv2d_cnhw: T.Buffer((T.int64(256), T.int64(128), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(128), T.int64(18), T.int64(18)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(128), T.int64(18), T.int64(18), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(128), i2)
                v_i2 = T.axis.spatial(T.int64(18), i3)
                v_i3 = T.axis.spatial(T.int64(18), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(17) and T.int64(1) <= v_i3 and v_i3 < T.int64(17), A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for ff, nn, yy, xx, rc, ry, rx, vi in T.grid(T.int64(256), T.int64(128), T.int64(4), T.int64(4), T.int64(32), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("conv2d_cnhw"):
                vi = T.axis.spatial(T.int64(1), ff)
                v_ff = T.axis.spatial(T.int64(256), nn)
                v_nn = T.axis.spatial(T.int64(128), yy)
                v_yy = T.axis.spatial(T.int64(4), xx)
                v_xx = T.axis.spatial(T.int64(4), rc)
                v_rc = T.axis.reduce(T.int64(32), ry)
                v_ry = T.axis.reduce(T.int64(8), rx)
                v_rx = T.axis.reduce(T.int64(8), T.int64(0))
                T.reads(pad_temp[v_rc, v_nn, v_ry * T.int64(2) + v_yy, v_rx * T.int64(2) + v_xx], B[v_rc, v_ff, v_ry, v_rx])
                T.writes(conv2d_cnhw[v_ff, v_nn, v_yy, v_xx])
                with T.init():
                    conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = T.float32(0)
                conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] + pad_temp[v_rc, v_nn, v_ry * T.int64(2) + v_yy, v_rx * T.int64(2) + v_xx] * B[v_rc, v_ff, v_ry, v_rx]

    @T.prim_func
    def conv2d17(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), B: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), conv2d_cnhw: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(128), T.int64(18), T.int64(18)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(128), T.int64(18), T.int64(18), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(128), i2)
                v_i2 = T.axis.spatial(T.int64(18), i3)
                v_i3 = T.axis.spatial(T.int64(18), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(17) and T.int64(1) <= v_i3 and v_i3 < T.int64(17), A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for ff, nn, yy, xx, rc, ry, rx, vi in T.grid(T.int64(128), T.int64(128), T.int64(3), T.int64(3), T.int64(32), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("conv2d_cnhw"):
                vi = T.axis.spatial(T.int64(1), ff)
                v_ff = T.axis.spatial(T.int64(128), nn)
                v_nn = T.axis.spatial(T.int64(128), yy)
                v_yy = T.axis.spatial(T.int64(3), xx)
                v_xx = T.axis.spatial(T.int64(3), rc)
                v_rc = T.axis.reduce(T.int64(32), ry)
                v_ry = T.axis.reduce(T.int64(16), rx)
                v_rx = T.axis.reduce(T.int64(16), T.int64(0))
                T.reads(pad_temp[v_rc, v_nn, v_yy + v_ry, v_xx + v_rx], B[v_rc, v_ff, v_ry, v_rx])
                T.writes(conv2d_cnhw[v_ff, v_nn, v_yy, v_xx])
                with T.init():
                    conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = T.float32(0)
                conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] + pad_temp[v_rc, v_nn, v_yy + v_ry, v_xx + v_rx] * B[v_rc, v_ff, v_ry, v_rx]

    @T.prim_func
    def conv2d18(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), B: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), conv2d_cnhw: T.Buffer((T.int64(128), T.int64(64), T.int64(2), T.int64(2)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(64), i2)
                v_i2 = T.axis.spatial(T.int64(32), i3)
                v_i3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, v_i2, v_i3]
        for ff, nn, yy, xx, rc, ry, rx, vi in T.grid(T.int64(128), T.int64(64), T.int64(2), T.int64(2), T.int64(32), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("conv2d_cnhw"):
                vi = T.axis.spatial(T.int64(1), ff)
                v_ff = T.axis.spatial(T.int64(128), nn)
                v_nn = T.axis.spatial(T.int64(64), yy)
                v_yy = T.axis.spatial(T.int64(2), xx)
                v_xx = T.axis.spatial(T.int64(2), rc)
                v_rc = T.axis.reduce(T.int64(32), ry)
                v_ry = T.axis.reduce(T.int64(16), rx)
                v_rx = T.axis.reduce(T.int64(16), T.int64(0))
                T.reads(pad_temp[v_rc, v_nn, v_ry * T.int64(2) + v_yy, v_rx * T.int64(2) + v_xx], B[v_rc, v_ff, v_ry, v_rx])
                T.writes(conv2d_cnhw[v_ff, v_nn, v_yy, v_xx])
                with T.init():
                    conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = T.float32(0)
                conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] + pad_temp[v_rc, v_nn, v_ry * T.int64(2) + v_yy, v_rx * T.int64(2) + v_xx] * B[v_rc, v_ff, v_ry, v_rx]

    @T.prim_func
    def conv2d19(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), B: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), conv2d_cnhw: T.Buffer((T.int64(128), T.int64(64), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(64), T.int64(34), T.int64(34)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(64), T.int64(34), T.int64(34), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(64), i2)
                v_i2 = T.axis.spatial(T.int64(34), i3)
                v_i3 = T.axis.spatial(T.int64(34), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(33) and T.int64(1) <= v_i3 and v_i3 < T.int64(33), A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for ff, nn, yy, xx, rc, ry, rx, vi in T.grid(T.int64(128), T.int64(64), T.int64(4), T.int64(4), T.int64(32), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("conv2d_cnhw"):
                vi = T.axis.spatial(T.int64(1), ff)
                v_ff = T.axis.spatial(T.int64(128), nn)
                v_nn = T.axis.spatial(T.int64(64), yy)
                v_yy = T.axis.spatial(T.int64(4), xx)
                v_xx = T.axis.spatial(T.int64(4), rc)
                v_rc = T.axis.reduce(T.int64(32), ry)
                v_ry = T.axis.reduce(T.int64(16), rx)
                v_rx = T.axis.reduce(T.int64(16), T.int64(0))
                T.reads(pad_temp[v_rc, v_nn, v_ry * T.int64(2) + v_yy, v_rx * T.int64(2) + v_xx], B[v_rc, v_ff, v_ry, v_rx])
                T.writes(conv2d_cnhw[v_ff, v_nn, v_yy, v_xx])
                with T.init():
                    conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = T.float32(0)
                conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] + pad_temp[v_rc, v_nn, v_ry * T.int64(2) + v_yy, v_rx * T.int64(2) + v_xx] * B[v_rc, v_ff, v_ry, v_rx]

    @T.prim_func
    def conv2d2(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), B: T.Buffer((T.int64(128), T.int64(64), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(64), T.int64(34), T.int64(34)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(64), T.int64(34), T.int64(34), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(64), i2)
                v_i2 = T.axis.spatial(T.int64(34), i3)
                v_i3 = T.axis.spatial(T.int64(34), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(33) and T.int64(1) <= v_i3 and v_i3 < T.int64(33), A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for nn, ff, yy, xx, rc, ry, rx, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(64), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("conv2d_nchw"):
                vi = T.axis.spatial(T.int64(1), nn)
                v_nn = T.axis.spatial(T.int64(32), ff)
                v_ff = T.axis.spatial(T.int64(128), yy)
                v_yy = T.axis.spatial(T.int64(16), xx)
                v_xx = T.axis.spatial(T.int64(16), rc)
                v_rc = T.axis.reduce(T.int64(64), ry)
                v_ry = T.axis.reduce(T.int64(3), rx)
                v_rx = T.axis.reduce(T.int64(3), T.int64(0))
                T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func
    def conv2d20(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), B: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), conv2d_cnhw: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(64), T.int64(34), T.int64(34)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(64), T.int64(34), T.int64(34), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(64), i2)
                v_i2 = T.axis.spatial(T.int64(34), i3)
                v_i3 = T.axis.spatial(T.int64(34), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(33) and T.int64(1) <= v_i3 and v_i3 < T.int64(33), A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for ff, nn, yy, xx, rc, ry, rx, vi in T.grid(T.int64(64), T.int64(64), T.int64(3), T.int64(3), T.int64(32), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("conv2d_cnhw"):
                vi = T.axis.spatial(T.int64(1), ff)
                v_ff = T.axis.spatial(T.int64(64), nn)
                v_nn = T.axis.spatial(T.int64(64), yy)
                v_yy = T.axis.spatial(T.int64(3), xx)
                v_xx = T.axis.spatial(T.int64(3), rc)
                v_rc, v_ry = T.axis.remap("RR", [ry, rx])
                v_rx = T.axis.reduce(T.int64(32), T.int64(0))
                T.reads(pad_temp[v_rc, v_nn, v_yy + v_ry, v_xx + v_rx], B[v_rc, v_ff, v_ry, v_rx])
                T.writes(conv2d_cnhw[v_ff, v_nn, v_yy, v_xx])
                with T.init():
                    conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = T.float32(0)
                conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] + pad_temp[v_rc, v_nn, v_yy + v_ry, v_xx + v_rx] * B[v_rc, v_ff, v_ry, v_rx]

    @T.prim_func
    def conv2d21(A: T.Buffer((T.int64(32), T.int64(3), T.int64(32), T.int64(32)), "float32"), B: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), conv2d_cnhw: T.Buffer((T.int64(64), T.int64(3), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(3), T.int64(34), T.int64(34)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(3), T.int64(34), T.int64(34), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(3), i2)
                v_i2 = T.axis.spatial(T.int64(34), i3)
                v_i3 = T.axis.spatial(T.int64(34), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(33) and T.int64(1) <= v_i3 and v_i3 < T.int64(33), A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for ff, nn, yy, xx, rc, ry, rx, vi in T.grid(T.int64(64), T.int64(3), T.int64(3), T.int64(3), T.int64(32), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("conv2d_cnhw"):
                vi = T.axis.spatial(T.int64(1), ff)
                v_ff = T.axis.spatial(T.int64(64), nn)
                v_nn, v_yy = T.axis.remap("SS", [yy, xx])
                v_xx = T.axis.spatial(T.int64(3), rc)
                v_rc, v_ry = T.axis.remap("RR", [ry, rx])
                v_rx = T.axis.reduce(T.int64(32), T.int64(0))
                T.reads(pad_temp[v_rc, v_nn, v_yy + v_ry, v_xx + v_rx], B[v_rc, v_ff, v_ry, v_rx])
                T.writes(conv2d_cnhw[v_ff, v_nn, v_yy, v_xx])
                with T.init():
                    conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = T.float32(0)
                conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] = conv2d_cnhw[v_ff, v_nn, v_yy, v_xx] + pad_temp[v_rc, v_nn, v_yy + v_ry, v_xx + v_rx] * B[v_rc, v_ff, v_ry, v_rx]

    @T.prim_func
    def conv2d3(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), B: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(128), T.int64(18), T.int64(18)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(128), T.int64(18), T.int64(18), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(128), i2)
                v_i2 = T.axis.spatial(T.int64(18), i3)
                v_i3 = T.axis.spatial(T.int64(18), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(17) and T.int64(1) <= v_i3 and v_i3 < T.int64(17), A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for nn, ff, yy, xx, rc, ry, rx, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(128), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("conv2d_nchw"):
                vi = T.axis.spatial(T.int64(1), nn)
                v_nn = T.axis.spatial(T.int64(32), ff)
                v_ff = T.axis.spatial(T.int64(128), yy)
                v_yy = T.axis.spatial(T.int64(16), xx)
                v_xx = T.axis.spatial(T.int64(16), rc)
                v_rc = T.axis.reduce(T.int64(128), ry)
                v_ry = T.axis.reduce(T.int64(3), rx)
                v_rx = T.axis.reduce(T.int64(3), T.int64(0))
                T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func
    def conv2d4(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), B: T.Buffer((T.int64(128), T.int64(64), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(64), i2)
                v_i2 = T.axis.spatial(T.int64(32), i3)
                v_i3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, v_i2, v_i3]
        for nn, ff, yy, xx, rc, ry, rx, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(64), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("conv2d_nchw"):
                vi = T.axis.spatial(T.int64(1), nn)
                v_nn = T.axis.spatial(T.int64(32), ff)
                v_ff = T.axis.spatial(T.int64(128), yy)
                v_yy = T.axis.spatial(T.int64(16), xx)
                v_xx = T.axis.spatial(T.int64(16), rc)
                v_rc = T.axis.reduce(T.int64(64), ry)
                v_ry = T.axis.reduce(T.int64(1), rx)
                v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func
    def conv2d5(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), B: T.Buffer((T.int64(256), T.int64(128), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(128), T.int64(18), T.int64(18)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(128), T.int64(18), T.int64(18), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(128), i2)
                v_i2 = T.axis.spatial(T.int64(18), i3)
                v_i3 = T.axis.spatial(T.int64(18), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(17) and T.int64(1) <= v_i3 and v_i3 < T.int64(17), A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for nn, ff, yy, xx, rc, ry, rx, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(128), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("conv2d_nchw"):
                vi = T.axis.spatial(T.int64(1), nn)
                v_nn = T.axis.spatial(T.int64(32), ff)
                v_ff = T.axis.spatial(T.int64(256), yy)
                v_yy = T.axis.spatial(T.int64(8), xx)
                v_xx = T.axis.spatial(T.int64(8), rc)
                v_rc = T.axis.reduce(T.int64(128), ry)
                v_ry = T.axis.reduce(T.int64(3), rx)
                v_rx = T.axis.reduce(T.int64(3), T.int64(0))
                T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func
    def conv2d6(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), B: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(256), T.int64(10), T.int64(10)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(256), T.int64(10), T.int64(10), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(256), i2)
                v_i2 = T.axis.spatial(T.int64(10), i3)
                v_i3 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(9) and T.int64(1) <= v_i3 and v_i3 < T.int64(9), A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for nn, ff, yy, xx, rc, ry, rx, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(256), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("conv2d_nchw"):
                vi = T.axis.spatial(T.int64(1), nn)
                v_nn = T.axis.spatial(T.int64(32), ff)
                v_ff = T.axis.spatial(T.int64(256), yy)
                v_yy = T.axis.spatial(T.int64(8), xx)
                v_xx = T.axis.spatial(T.int64(8), rc)
                v_rc = T.axis.reduce(T.int64(256), ry)
                v_ry = T.axis.reduce(T.int64(3), rx)
                v_rx = T.axis.reduce(T.int64(3), T.int64(0))
                T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func
    def conv2d7(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), B: T.Buffer((T.int64(256), T.int64(128), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(128), i2)
                v_i2 = T.axis.spatial(T.int64(16), i3)
                v_i3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, v_i2, v_i3]
        for nn, ff, yy, xx, rc, ry, rx, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(128), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("conv2d_nchw"):
                vi = T.axis.spatial(T.int64(1), nn)
                v_nn = T.axis.spatial(T.int64(32), ff)
                v_ff = T.axis.spatial(T.int64(256), yy)
                v_yy = T.axis.spatial(T.int64(8), xx)
                v_xx = T.axis.spatial(T.int64(8), rc)
                v_rc = T.axis.reduce(T.int64(128), ry)
                v_ry = T.axis.reduce(T.int64(1), rx)
                v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func
    def conv2d8(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), B: T.Buffer((T.int64(512), T.int64(256), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(256), T.int64(10), T.int64(10)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(256), T.int64(10), T.int64(10), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(256), i2)
                v_i2 = T.axis.spatial(T.int64(10), i3)
                v_i3 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(9) and T.int64(1) <= v_i3 and v_i3 < T.int64(9), A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for nn, ff, yy, xx, rc, ry, rx, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(256), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("conv2d_nchw"):
                vi = T.axis.spatial(T.int64(1), nn)
                v_nn = T.axis.spatial(T.int64(32), ff)
                v_ff = T.axis.spatial(T.int64(512), yy)
                v_yy = T.axis.spatial(T.int64(4), xx)
                v_xx = T.axis.spatial(T.int64(4), rc)
                v_rc = T.axis.reduce(T.int64(256), ry)
                v_ry = T.axis.reduce(T.int64(3), rx)
                v_rx = T.axis.reduce(T.int64(3), T.int64(0))
                T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func
    def conv2d9(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), B: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(512), T.int64(6), T.int64(6)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(512), T.int64(6), T.int64(6), T.int64(1)):
            with T.block("pad_temp"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(512), i2)
                v_i2 = T.axis.spatial(T.int64(6), i3)
                v_i3 = T.axis.spatial(T.int64(6), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(5) and T.int64(1) <= v_i3 and v_i3 < T.int64(5), A[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for nn, ff, yy, xx, rc, ry, rx, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(512), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("conv2d_nchw"):
                vi = T.axis.spatial(T.int64(1), nn)
                v_nn = T.axis.spatial(T.int64(32), ff)
                v_ff = T.axis.spatial(T.int64(512), yy)
                v_yy = T.axis.spatial(T.int64(4), xx)
                v_xx = T.axis.spatial(T.int64(4), rc)
                v_rc = T.axis.reduce(T.int64(512), ry)
                v_ry = T.axis.reduce(T.int64(3), rx)
                v_rx = T.axis.reduce(T.int64(3), T.int64(0))
                T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func
    def conv2d_transpose(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), B: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), compute: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        data_dilate = T.alloc_buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)))
        data_pad = T.alloc_buffer((T.int64(32), T.int64(512), T.int64(6), T.int64(6)))
        kernel_transform = T.alloc_buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("data_dilate"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(512), i2)
                v_i2 = T.axis.spatial(T.int64(4), i3)
                v_i3 = T.axis.spatial(T.int64(4), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(data_dilate[v_i0, v_i1, v_i2, v_i3])
                data_dilate[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, v_i2, v_i3]
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(512), T.int64(6), T.int64(6), T.int64(1)):
            with T.block("data_pad"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(512), i2)
                v_i2 = T.axis.spatial(T.int64(6), i3)
                v_i3 = T.axis.spatial(T.int64(6), T.int64(0))
                T.reads(data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(data_pad[v_i0, v_i1, v_i2, v_i3])
                data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(5) and T.int64(1) <= v_i3 and v_i3 < T.int64(5), data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for o, i, h, w, vi in T.grid(T.int64(512), T.int64(512), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("kernel_transform"):
                vi = T.axis.spatial(T.int64(1), o)
                v_o = T.axis.spatial(T.int64(512), i)
                v_i = T.axis.spatial(T.int64(512), h)
                v_h = T.axis.spatial(T.int64(3), w)
                v_w = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(B[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w])
                T.writes(kernel_transform[v_o, v_i, v_h, v_w])
                kernel_transform[v_o, v_i, v_h, v_w] = B[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w]
        for b, c, h, w, dc, dh, dw, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(512), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), b)
                v_b = T.axis.spatial(T.int64(32), c)
                v_c = T.axis.spatial(T.int64(512), h)
                v_h = T.axis.spatial(T.int64(4), w)
                v_w = T.axis.spatial(T.int64(4), dc)
                v_dc = T.axis.reduce(T.int64(512), dh)
                v_dh = T.axis.reduce(T.int64(3), dw)
                v_dw = T.axis.reduce(T.int64(3), T.int64(0))
                T.reads(data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw], kernel_transform[v_c, v_dc, v_dh, v_dw])
                T.writes(compute[v_b, v_c, v_h, v_w])
                with T.init():
                    compute[v_b, v_c, v_h, v_w] = T.float32(0)
                compute[v_b, v_c, v_h, v_w] = compute[v_b, v_c, v_h, v_w] + data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw] * kernel_transform[v_c, v_dc, v_dh, v_dw]

    @T.prim_func
    def conv2d_transpose1(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), B: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32"), compute: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        data_dilate = T.alloc_buffer((T.int64(32), T.int64(512), T.int64(7), T.int64(7)))
        data_pad = T.alloc_buffer((T.int64(32), T.int64(512), T.int64(8), T.int64(8)))
        kernel_transform = T.alloc_buffer((T.int64(256), T.int64(512), T.int64(1), T.int64(1)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(512), T.int64(7), T.int64(7), T.int64(1)):
            with T.block("data_dilate"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(512), i2)
                v_i2 = T.axis.spatial(T.int64(7), i3)
                v_i3 = T.axis.spatial(T.int64(7), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 // T.int64(2), v_i3 // T.int64(2)])
                T.writes(data_dilate[v_i0, v_i1, v_i2, v_i3])
                data_dilate[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(v_i2 % T.int64(2) == T.int64(0) and v_i3 % T.int64(2) == T.int64(0), A[v_i0, v_i1, v_i2 // T.int64(2), v_i3 // T.int64(2)], T.float32(0))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(512), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("data_pad"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(512), i2)
                v_i2 = T.axis.spatial(T.int64(8), i3)
                v_i3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(data_dilate[v_i0, v_i1, v_i2, v_i3])
                T.writes(data_pad[v_i0, v_i1, v_i2, v_i3])
                data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(0) <= v_i2 and v_i2 < T.int64(7) and T.int64(0) <= v_i3 and v_i3 < T.int64(7), data_dilate[v_i0, v_i1, v_i2, v_i3], T.float32(0))
        for o, i, h, w, vi in T.grid(T.int64(256), T.int64(512), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("kernel_transform"):
                vi = T.axis.spatial(T.int64(1), o)
                v_o = T.axis.spatial(T.int64(256), i)
                v_i = T.axis.spatial(T.int64(512), h)
                v_h = T.axis.spatial(T.int64(1), w)
                v_w = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(B[v_i, v_o, T.int64(0) - v_h, T.int64(0) - v_w])
                T.writes(kernel_transform[v_o, v_i, v_h, v_w])
                kernel_transform[v_o, v_i, v_h, v_w] = B[v_i, v_o, T.int64(0) - v_h, T.int64(0) - v_w]
        for b, c, h, w, dc, dh, dw, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(512), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), b)
                v_b = T.axis.spatial(T.int64(32), c)
                v_c = T.axis.spatial(T.int64(256), h)
                v_h = T.axis.spatial(T.int64(8), w)
                v_w = T.axis.spatial(T.int64(8), dc)
                v_dc = T.axis.reduce(T.int64(512), dh)
                v_dh = T.axis.reduce(T.int64(1), dw)
                v_dw = T.axis.reduce(T.int64(1), T.int64(0))
                T.reads(data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw], kernel_transform[v_c, v_dc, v_dh, v_dw])
                T.writes(compute[v_b, v_c, v_h, v_w])
                with T.init():
                    compute[v_b, v_c, v_h, v_w] = T.float32(0)
                compute[v_b, v_c, v_h, v_w] = compute[v_b, v_c, v_h, v_w] + data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw] * kernel_transform[v_c, v_dc, v_dh, v_dw]

    @T.prim_func
    def conv2d_transpose2(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), B: T.Buffer((T.int64(512), T.int64(256), T.int64(3), T.int64(3)), "float32"), compute: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        data_dilate = T.alloc_buffer((T.int64(32), T.int64(512), T.int64(7), T.int64(7)))
        data_pad = T.alloc_buffer((T.int64(32), T.int64(512), T.int64(10), T.int64(10)))
        kernel_transform = T.alloc_buffer((T.int64(256), T.int64(512), T.int64(3), T.int64(3)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(512), T.int64(7), T.int64(7), T.int64(1)):
            with T.block("data_dilate"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(512), i2)
                v_i2 = T.axis.spatial(T.int64(7), i3)
                v_i3 = T.axis.spatial(T.int64(7), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 // T.int64(2), v_i3 // T.int64(2)])
                T.writes(data_dilate[v_i0, v_i1, v_i2, v_i3])
                data_dilate[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(v_i2 % T.int64(2) == T.int64(0) and v_i3 % T.int64(2) == T.int64(0), A[v_i0, v_i1, v_i2 // T.int64(2), v_i3 // T.int64(2)], T.float32(0))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(512), T.int64(10), T.int64(10), T.int64(1)):
            with T.block("data_pad"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(512), i2)
                v_i2 = T.axis.spatial(T.int64(10), i3)
                v_i3 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(data_pad[v_i0, v_i1, v_i2, v_i3])
                data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(8) and T.int64(1) <= v_i3 and v_i3 < T.int64(8), data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for o, i, h, w, vi in T.grid(T.int64(256), T.int64(512), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("kernel_transform"):
                vi = T.axis.spatial(T.int64(1), o)
                v_o = T.axis.spatial(T.int64(256), i)
                v_i = T.axis.spatial(T.int64(512), h)
                v_h = T.axis.spatial(T.int64(3), w)
                v_w = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(B[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w])
                T.writes(kernel_transform[v_o, v_i, v_h, v_w])
                kernel_transform[v_o, v_i, v_h, v_w] = B[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w]
        for b, c, h, w, dc, dh, dw, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(512), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), b)
                v_b = T.axis.spatial(T.int64(32), c)
                v_c = T.axis.spatial(T.int64(256), h)
                v_h = T.axis.spatial(T.int64(8), w)
                v_w = T.axis.spatial(T.int64(8), dc)
                v_dc = T.axis.reduce(T.int64(512), dh)
                v_dh = T.axis.reduce(T.int64(3), dw)
                v_dw = T.axis.reduce(T.int64(3), T.int64(0))
                T.reads(data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw], kernel_transform[v_c, v_dc, v_dh, v_dw])
                T.writes(compute[v_b, v_c, v_h, v_w])
                with T.init():
                    compute[v_b, v_c, v_h, v_w] = T.float32(0)
                compute[v_b, v_c, v_h, v_w] = compute[v_b, v_c, v_h, v_w] + data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw] * kernel_transform[v_c, v_dc, v_dh, v_dw]

    @T.prim_func
    def conv2d_transpose3(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), B: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), compute: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        data_dilate = T.alloc_buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)))
        data_pad = T.alloc_buffer((T.int64(32), T.int64(256), T.int64(10), T.int64(10)))
        kernel_transform = T.alloc_buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("data_dilate"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(256), i2)
                v_i2 = T.axis.spatial(T.int64(8), i3)
                v_i3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(data_dilate[v_i0, v_i1, v_i2, v_i3])
                data_dilate[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, v_i2, v_i3]
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(256), T.int64(10), T.int64(10), T.int64(1)):
            with T.block("data_pad"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(256), i2)
                v_i2 = T.axis.spatial(T.int64(10), i3)
                v_i3 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(data_pad[v_i0, v_i1, v_i2, v_i3])
                data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(9) and T.int64(1) <= v_i3 and v_i3 < T.int64(9), data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for o, i, h, w, vi in T.grid(T.int64(256), T.int64(256), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("kernel_transform"):
                vi = T.axis.spatial(T.int64(1), o)
                v_o = T.axis.spatial(T.int64(256), i)
                v_i = T.axis.spatial(T.int64(256), h)
                v_h = T.axis.spatial(T.int64(3), w)
                v_w = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(B[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w])
                T.writes(kernel_transform[v_o, v_i, v_h, v_w])
                kernel_transform[v_o, v_i, v_h, v_w] = B[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w]
        for b, c, h, w, dc, dh, dw, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(256), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), b)
                v_b = T.axis.spatial(T.int64(32), c)
                v_c = T.axis.spatial(T.int64(256), h)
                v_h = T.axis.spatial(T.int64(8), w)
                v_w = T.axis.spatial(T.int64(8), dc)
                v_dc = T.axis.reduce(T.int64(256), dh)
                v_dh = T.axis.reduce(T.int64(3), dw)
                v_dw = T.axis.reduce(T.int64(3), T.int64(0))
                T.reads(data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw], kernel_transform[v_c, v_dc, v_dh, v_dw])
                T.writes(compute[v_b, v_c, v_h, v_w])
                with T.init():
                    compute[v_b, v_c, v_h, v_w] = T.float32(0)
                compute[v_b, v_c, v_h, v_w] = compute[v_b, v_c, v_h, v_w] + data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw] * kernel_transform[v_c, v_dc, v_dh, v_dw]

    @T.prim_func
    def conv2d_transpose4(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), B: T.Buffer((T.int64(256), T.int64(128), T.int64(1), T.int64(1)), "float32"), compute: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        data_dilate = T.alloc_buffer((T.int64(32), T.int64(256), T.int64(15), T.int64(15)))
        data_pad = T.alloc_buffer((T.int64(32), T.int64(256), T.int64(16), T.int64(16)))
        kernel_transform = T.alloc_buffer((T.int64(128), T.int64(256), T.int64(1), T.int64(1)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(256), T.int64(15), T.int64(15), T.int64(1)):
            with T.block("data_dilate"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(256), i2)
                v_i2 = T.axis.spatial(T.int64(15), i3)
                v_i3 = T.axis.spatial(T.int64(15), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 // T.int64(2), v_i3 // T.int64(2)])
                T.writes(data_dilate[v_i0, v_i1, v_i2, v_i3])
                data_dilate[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(v_i2 % T.int64(2) == T.int64(0) and v_i3 % T.int64(2) == T.int64(0), A[v_i0, v_i1, v_i2 // T.int64(2), v_i3 // T.int64(2)], T.float32(0))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(256), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("data_pad"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(256), i2)
                v_i2 = T.axis.spatial(T.int64(16), i3)
                v_i3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(data_dilate[v_i0, v_i1, v_i2, v_i3])
                T.writes(data_pad[v_i0, v_i1, v_i2, v_i3])
                data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(0) <= v_i2 and v_i2 < T.int64(15) and T.int64(0) <= v_i3 and v_i3 < T.int64(15), data_dilate[v_i0, v_i1, v_i2, v_i3], T.float32(0))
        for o, i, h, w, vi in T.grid(T.int64(128), T.int64(256), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("kernel_transform"):
                vi = T.axis.spatial(T.int64(1), o)
                v_o = T.axis.spatial(T.int64(128), i)
                v_i = T.axis.spatial(T.int64(256), h)
                v_h = T.axis.spatial(T.int64(1), w)
                v_w = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(B[v_i, v_o, T.int64(0) - v_h, T.int64(0) - v_w])
                T.writes(kernel_transform[v_o, v_i, v_h, v_w])
                kernel_transform[v_o, v_i, v_h, v_w] = B[v_i, v_o, T.int64(0) - v_h, T.int64(0) - v_w]
        for b, c, h, w, dc, dh, dw, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(256), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), b)
                v_b = T.axis.spatial(T.int64(32), c)
                v_c = T.axis.spatial(T.int64(128), h)
                v_h = T.axis.spatial(T.int64(16), w)
                v_w = T.axis.spatial(T.int64(16), dc)
                v_dc = T.axis.reduce(T.int64(256), dh)
                v_dh = T.axis.reduce(T.int64(1), dw)
                v_dw = T.axis.reduce(T.int64(1), T.int64(0))
                T.reads(data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw], kernel_transform[v_c, v_dc, v_dh, v_dw])
                T.writes(compute[v_b, v_c, v_h, v_w])
                with T.init():
                    compute[v_b, v_c, v_h, v_w] = T.float32(0)
                compute[v_b, v_c, v_h, v_w] = compute[v_b, v_c, v_h, v_w] + data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw] * kernel_transform[v_c, v_dc, v_dh, v_dw]

    @T.prim_func
    def conv2d_transpose5(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), B: T.Buffer((T.int64(256), T.int64(128), T.int64(3), T.int64(3)), "float32"), compute: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        data_dilate = T.alloc_buffer((T.int64(32), T.int64(256), T.int64(15), T.int64(15)))
        data_pad = T.alloc_buffer((T.int64(32), T.int64(256), T.int64(18), T.int64(18)))
        kernel_transform = T.alloc_buffer((T.int64(128), T.int64(256), T.int64(3), T.int64(3)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(256), T.int64(15), T.int64(15), T.int64(1)):
            with T.block("data_dilate"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(256), i2)
                v_i2 = T.axis.spatial(T.int64(15), i3)
                v_i3 = T.axis.spatial(T.int64(15), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 // T.int64(2), v_i3 // T.int64(2)])
                T.writes(data_dilate[v_i0, v_i1, v_i2, v_i3])
                data_dilate[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(v_i2 % T.int64(2) == T.int64(0) and v_i3 % T.int64(2) == T.int64(0), A[v_i0, v_i1, v_i2 // T.int64(2), v_i3 // T.int64(2)], T.float32(0))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(256), T.int64(18), T.int64(18), T.int64(1)):
            with T.block("data_pad"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(256), i2)
                v_i2 = T.axis.spatial(T.int64(18), i3)
                v_i3 = T.axis.spatial(T.int64(18), T.int64(0))
                T.reads(data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(data_pad[v_i0, v_i1, v_i2, v_i3])
                data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(16) and T.int64(1) <= v_i3 and v_i3 < T.int64(16), data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for o, i, h, w, vi in T.grid(T.int64(128), T.int64(256), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("kernel_transform"):
                vi = T.axis.spatial(T.int64(1), o)
                v_o = T.axis.spatial(T.int64(128), i)
                v_i = T.axis.spatial(T.int64(256), h)
                v_h = T.axis.spatial(T.int64(3), w)
                v_w = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(B[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w])
                T.writes(kernel_transform[v_o, v_i, v_h, v_w])
                kernel_transform[v_o, v_i, v_h, v_w] = B[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w]
        for b, c, h, w, dc, dh, dw, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(256), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), b)
                v_b = T.axis.spatial(T.int64(32), c)
                v_c = T.axis.spatial(T.int64(128), h)
                v_h = T.axis.spatial(T.int64(16), w)
                v_w = T.axis.spatial(T.int64(16), dc)
                v_dc = T.axis.reduce(T.int64(256), dh)
                v_dh = T.axis.reduce(T.int64(3), dw)
                v_dw = T.axis.reduce(T.int64(3), T.int64(0))
                T.reads(data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw], kernel_transform[v_c, v_dc, v_dh, v_dw])
                T.writes(compute[v_b, v_c, v_h, v_w])
                with T.init():
                    compute[v_b, v_c, v_h, v_w] = T.float32(0)
                compute[v_b, v_c, v_h, v_w] = compute[v_b, v_c, v_h, v_w] + data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw] * kernel_transform[v_c, v_dc, v_dh, v_dw]

    @T.prim_func
    def conv2d_transpose6(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), B: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32"), compute: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        data_dilate = T.alloc_buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)))
        data_pad = T.alloc_buffer((T.int64(32), T.int64(128), T.int64(18), T.int64(18)))
        kernel_transform = T.alloc_buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("data_dilate"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(128), i2)
                v_i2 = T.axis.spatial(T.int64(16), i3)
                v_i3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(data_dilate[v_i0, v_i1, v_i2, v_i3])
                data_dilate[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, v_i2, v_i3]
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(128), T.int64(18), T.int64(18), T.int64(1)):
            with T.block("data_pad"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(128), i2)
                v_i2 = T.axis.spatial(T.int64(18), i3)
                v_i3 = T.axis.spatial(T.int64(18), T.int64(0))
                T.reads(data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(data_pad[v_i0, v_i1, v_i2, v_i3])
                data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(17) and T.int64(1) <= v_i3 and v_i3 < T.int64(17), data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for o, i, h, w, vi in T.grid(T.int64(128), T.int64(128), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("kernel_transform"):
                vi = T.axis.spatial(T.int64(1), o)
                v_o = T.axis.spatial(T.int64(128), i)
                v_i = T.axis.spatial(T.int64(128), h)
                v_h = T.axis.spatial(T.int64(3), w)
                v_w = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(B[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w])
                T.writes(kernel_transform[v_o, v_i, v_h, v_w])
                kernel_transform[v_o, v_i, v_h, v_w] = B[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w]
        for b, c, h, w, dc, dh, dw, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(128), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), b)
                v_b = T.axis.spatial(T.int64(32), c)
                v_c = T.axis.spatial(T.int64(128), h)
                v_h = T.axis.spatial(T.int64(16), w)
                v_w = T.axis.spatial(T.int64(16), dc)
                v_dc = T.axis.reduce(T.int64(128), dh)
                v_dh = T.axis.reduce(T.int64(3), dw)
                v_dw = T.axis.reduce(T.int64(3), T.int64(0))
                T.reads(data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw], kernel_transform[v_c, v_dc, v_dh, v_dw])
                T.writes(compute[v_b, v_c, v_h, v_w])
                with T.init():
                    compute[v_b, v_c, v_h, v_w] = T.float32(0)
                compute[v_b, v_c, v_h, v_w] = compute[v_b, v_c, v_h, v_w] + data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw] * kernel_transform[v_c, v_dc, v_dh, v_dw]

    @T.prim_func
    def conv2d_transpose7(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), B: T.Buffer((T.int64(128), T.int64(64), T.int64(1), T.int64(1)), "float32"), compute: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        data_dilate = T.alloc_buffer((T.int64(32), T.int64(128), T.int64(31), T.int64(31)))
        data_pad = T.alloc_buffer((T.int64(32), T.int64(128), T.int64(32), T.int64(32)))
        kernel_transform = T.alloc_buffer((T.int64(64), T.int64(128), T.int64(1), T.int64(1)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(128), T.int64(31), T.int64(31), T.int64(1)):
            with T.block("data_dilate"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(128), i2)
                v_i2 = T.axis.spatial(T.int64(31), i3)
                v_i3 = T.axis.spatial(T.int64(31), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 // T.int64(2), v_i3 // T.int64(2)])
                T.writes(data_dilate[v_i0, v_i1, v_i2, v_i3])
                data_dilate[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(v_i2 % T.int64(2) == T.int64(0) and v_i3 % T.int64(2) == T.int64(0), A[v_i0, v_i1, v_i2 // T.int64(2), v_i3 // T.int64(2)], T.float32(0))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(128), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("data_pad"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(128), i2)
                v_i2 = T.axis.spatial(T.int64(32), i3)
                v_i3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(data_dilate[v_i0, v_i1, v_i2, v_i3])
                T.writes(data_pad[v_i0, v_i1, v_i2, v_i3])
                data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(0) <= v_i2 and v_i2 < T.int64(31) and T.int64(0) <= v_i3 and v_i3 < T.int64(31), data_dilate[v_i0, v_i1, v_i2, v_i3], T.float32(0))
        for o, i, h, w, vi in T.grid(T.int64(64), T.int64(128), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("kernel_transform"):
                vi = T.axis.spatial(T.int64(1), o)
                v_o = T.axis.spatial(T.int64(64), i)
                v_i = T.axis.spatial(T.int64(128), h)
                v_h = T.axis.spatial(T.int64(1), w)
                v_w = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(B[v_i, v_o, T.int64(0) - v_h, T.int64(0) - v_w])
                T.writes(kernel_transform[v_o, v_i, v_h, v_w])
                kernel_transform[v_o, v_i, v_h, v_w] = B[v_i, v_o, T.int64(0) - v_h, T.int64(0) - v_w]
        for b, c, h, w, dc, dh, dw, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(128), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), b)
                v_b = T.axis.spatial(T.int64(32), c)
                v_c = T.axis.spatial(T.int64(64), h)
                v_h = T.axis.spatial(T.int64(32), w)
                v_w = T.axis.spatial(T.int64(32), dc)
                v_dc = T.axis.reduce(T.int64(128), dh)
                v_dh = T.axis.reduce(T.int64(1), dw)
                v_dw = T.axis.reduce(T.int64(1), T.int64(0))
                T.reads(data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw], kernel_transform[v_c, v_dc, v_dh, v_dw])
                T.writes(compute[v_b, v_c, v_h, v_w])
                with T.init():
                    compute[v_b, v_c, v_h, v_w] = T.float32(0)
                compute[v_b, v_c, v_h, v_w] = compute[v_b, v_c, v_h, v_w] + data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw] * kernel_transform[v_c, v_dc, v_dh, v_dw]

    @T.prim_func
    def conv2d_transpose8(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), B: T.Buffer((T.int64(128), T.int64(64), T.int64(3), T.int64(3)), "float32"), compute: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        data_dilate = T.alloc_buffer((T.int64(32), T.int64(128), T.int64(31), T.int64(31)))
        data_pad = T.alloc_buffer((T.int64(32), T.int64(128), T.int64(34), T.int64(34)))
        kernel_transform = T.alloc_buffer((T.int64(64), T.int64(128), T.int64(3), T.int64(3)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(128), T.int64(31), T.int64(31), T.int64(1)):
            with T.block("data_dilate"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(128), i2)
                v_i2 = T.axis.spatial(T.int64(31), i3)
                v_i3 = T.axis.spatial(T.int64(31), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2 // T.int64(2), v_i3 // T.int64(2)])
                T.writes(data_dilate[v_i0, v_i1, v_i2, v_i3])
                data_dilate[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(v_i2 % T.int64(2) == T.int64(0) and v_i3 % T.int64(2) == T.int64(0), A[v_i0, v_i1, v_i2 // T.int64(2), v_i3 // T.int64(2)], T.float32(0))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(128), T.int64(34), T.int64(34), T.int64(1)):
            with T.block("data_pad"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(128), i2)
                v_i2 = T.axis.spatial(T.int64(34), i3)
                v_i3 = T.axis.spatial(T.int64(34), T.int64(0))
                T.reads(data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(data_pad[v_i0, v_i1, v_i2, v_i3])
                data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(32) and T.int64(1) <= v_i3 and v_i3 < T.int64(32), data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for o, i, h, w, vi in T.grid(T.int64(64), T.int64(128), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("kernel_transform"):
                vi = T.axis.spatial(T.int64(1), o)
                v_o = T.axis.spatial(T.int64(64), i)
                v_i = T.axis.spatial(T.int64(128), h)
                v_h = T.axis.spatial(T.int64(3), w)
                v_w = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(B[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w])
                T.writes(kernel_transform[v_o, v_i, v_h, v_w])
                kernel_transform[v_o, v_i, v_h, v_w] = B[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w]
        for b, c, h, w, dc, dh, dw, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(128), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), b)
                v_b = T.axis.spatial(T.int64(32), c)
                v_c = T.axis.spatial(T.int64(64), h)
                v_h = T.axis.spatial(T.int64(32), w)
                v_w = T.axis.spatial(T.int64(32), dc)
                v_dc = T.axis.reduce(T.int64(128), dh)
                v_dh = T.axis.reduce(T.int64(3), dw)
                v_dw = T.axis.reduce(T.int64(3), T.int64(0))
                T.reads(data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw], kernel_transform[v_c, v_dc, v_dh, v_dw])
                T.writes(compute[v_b, v_c, v_h, v_w])
                with T.init():
                    compute[v_b, v_c, v_h, v_w] = T.float32(0)
                compute[v_b, v_c, v_h, v_w] = compute[v_b, v_c, v_h, v_w] + data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw] * kernel_transform[v_c, v_dc, v_dh, v_dw]

    @T.prim_func
    def conv2d_transpose9(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), B: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32"), compute: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        data_dilate = T.alloc_buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)))
        data_pad = T.alloc_buffer((T.int64(32), T.int64(64), T.int64(34), T.int64(34)))
        kernel_transform = T.alloc_buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)))
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("data_dilate"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(64), i2)
                v_i2 = T.axis.spatial(T.int64(32), i3)
                v_i3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(data_dilate[v_i0, v_i1, v_i2, v_i3])
                data_dilate[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, v_i2, v_i3]
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(64), T.int64(34), T.int64(34), T.int64(1)):
            with T.block("data_pad"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(64), i2)
                v_i2 = T.axis.spatial(T.int64(34), i3)
                v_i3 = T.axis.spatial(T.int64(34), T.int64(0))
                T.reads(data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(data_pad[v_i0, v_i1, v_i2, v_i3])
                data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(33) and T.int64(1) <= v_i3 and v_i3 < T.int64(33), data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for o, i, h, w, vi in T.grid(T.int64(64), T.int64(64), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("kernel_transform"):
                vi = T.axis.spatial(T.int64(1), o)
                v_o = T.axis.spatial(T.int64(64), i)
                v_i = T.axis.spatial(T.int64(64), h)
                v_h = T.axis.spatial(T.int64(3), w)
                v_w = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(B[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w])
                T.writes(kernel_transform[v_o, v_i, v_h, v_w])
                kernel_transform[v_o, v_i, v_h, v_w] = B[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w]
        for b, c, h, w, dc, dh, dw, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(64), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), b)
                v_b = T.axis.spatial(T.int64(32), c)
                v_c = T.axis.spatial(T.int64(64), h)
                v_h = T.axis.spatial(T.int64(32), w)
                v_w = T.axis.spatial(T.int64(32), dc)
                v_dc = T.axis.reduce(T.int64(64), dh)
                v_dh = T.axis.reduce(T.int64(3), dw)
                v_dw = T.axis.reduce(T.int64(3), T.int64(0))
                T.reads(data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw], kernel_transform[v_c, v_dc, v_dh, v_dw])
                T.writes(compute[v_b, v_c, v_h, v_w])
                with T.init():
                    compute[v_b, v_c, v_h, v_w] = T.float32(0)
                compute[v_b, v_c, v_h, v_w] = compute[v_b, v_c, v_h, v_w] + data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw] * kernel_transform[v_c, v_dc, v_dh, v_dw]

    @T.prim_func
    def divide(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), B: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32"), T_divide: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(32), ax3)
                v_ax3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] / B[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def divide1(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), B: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32"), T_divide: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(16), ax3)
                v_ax3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] / B[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def divide10(A: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32"), B: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32"), T_divide: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(64), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] / B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def divide11(A: T.Buffer((T.int64(64),), "float32"), T_divide: T.Buffer((T.int64(64),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(64), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = A[v_ax0] * T.float32(3.0517578125e-05)

    @T.prim_func
    def divide2(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), B: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), T_divide: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(8), ax3)
                v_ax3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] / B[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def divide3(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), B: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), T_divide: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(4), ax3)
                v_ax3 = T.axis.spatial(T.int64(4), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] / B[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def divide4(A: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), B: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), T_divide: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(512), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] / B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def divide5(A: T.Buffer((T.int64(512),), "float32"), T_divide: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(512), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = A[v_ax0] * T.float32(0.001953125)

    @T.prim_func
    def divide6(A: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), B: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), T_divide: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(256), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] / B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def divide7(A: T.Buffer((T.int64(256),), "float32"), T_divide: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(256), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = A[v_ax0] * T.float32(0.00048828125)

    @T.prim_func
    def divide8(A: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32"), B: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32"), T_divide: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(128), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] / B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def divide9(A: T.Buffer((T.int64(128),), "float32"), T_divide: T.Buffer((T.int64(128),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(128), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = A[v_ax0] * T.float32(0.0001220703125)

    @T.prim_func
    def expand_dims(A: T.Buffer((T.int64(64),), "float32"), expand_dims_1: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, vi in T.grid(T.int64(1), T.int64(64), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("expand_dims"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(1), i1)
                v_i1 = T.axis.spatial(T.int64(64), i2)
                v_i2 = T.axis.spatial(T.int64(1), i3)
                v_i3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_i1])
                T.writes(expand_dims_1[v_i0, v_i1, v_i2, v_i3])
                expand_dims_1[v_i0, v_i1, v_i2, v_i3] = A[v_i1]

    @T.prim_func
    def expand_dims1(A: T.Buffer((T.int64(128),), "float32"), expand_dims: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, vi in T.grid(T.int64(1), T.int64(128), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("expand_dims"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(1), i1)
                v_i1 = T.axis.spatial(T.int64(128), i2)
                v_i2 = T.axis.spatial(T.int64(1), i3)
                v_i3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_i1])
                T.writes(expand_dims[v_i0, v_i1, v_i2, v_i3])
                expand_dims[v_i0, v_i1, v_i2, v_i3] = A[v_i1]

    @T.prim_func
    def expand_dims2(A: T.Buffer((T.int64(256),), "float32"), expand_dims: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, vi in T.grid(T.int64(1), T.int64(256), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("expand_dims"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(1), i1)
                v_i1 = T.axis.spatial(T.int64(256), i2)
                v_i2 = T.axis.spatial(T.int64(1), i3)
                v_i3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_i1])
                T.writes(expand_dims[v_i0, v_i1, v_i2, v_i3])
                expand_dims[v_i0, v_i1, v_i2, v_i3] = A[v_i1]

    @T.prim_func
    def expand_dims3(A: T.Buffer((T.int64(512),), "float32"), expand_dims: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, vi in T.grid(T.int64(1), T.int64(512), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("expand_dims"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(1), i1)
                v_i1 = T.axis.spatial(T.int64(512), i2)
                v_i2 = T.axis.spatial(T.int64(1), i3)
                v_i3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_i1])
                T.writes(expand_dims[v_i0, v_i1, v_i2, v_i3])
                expand_dims[v_i0, v_i1, v_i2, v_i3] = A[v_i1]

    @T.prim_func
    def less(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), T_less: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "bool")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("T_less"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(4), ax3)
                v_ax3 = T.axis.spatial(T.int64(4), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_less[v_ax0, v_ax1, v_ax2, v_ax3])
                T_less[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] < T.float32(0)

    @T.prim_func
    def less1(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), T_less: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "bool")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("T_less"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(8), ax3)
                v_ax3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_less[v_ax0, v_ax1, v_ax2, v_ax3])
                T_less[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] < T.float32(0)

    @T.prim_func
    def less2(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), T_less: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "bool")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("T_less"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(16), ax3)
                v_ax3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_less[v_ax0, v_ax1, v_ax2, v_ax3])
                T_less[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] < T.float32(0)

    @T.prim_func
    def less3(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), T_less: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "bool")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("T_less"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(32), ax3)
                v_ax3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_less[v_ax0, v_ax1, v_ax2, v_ax3])
                T_less[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] < T.float32(0)

    @T.prim_func
    def log_softmax(A: T.Buffer((T.int64(32), T.int64(10)), "float32"), compute: T.Buffer((T.int64(32), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(32),))
        compute_1 = T.alloc_buffer((T.int64(32),))
        for i0, k, vi in T.grid(T.int64(32), T.int64(10), T.int64(1)):
            with T.block("T_softmax_maxelem"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), k)
                v_k = T.axis.reduce(T.int64(10), T.int64(0))
                T.reads(A[v_i0, v_k])
                T.writes(T_softmax_maxelem[v_i0])
                with T.init():
                    T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], A[v_i0, v_k])
        for i0, k, vi in T.grid(T.int64(32), T.int64(10), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), k)
                v_k = T.axis.reduce(T.int64(10), T.int64(0))
                T.reads(A[v_i0, v_k], T_softmax_maxelem[v_i0])
                T.writes(compute_1[v_i0])
                with T.init():
                    compute_1[v_i0] = T.float32(0)
                compute_1[v_i0] = compute_1[v_i0] + T.exp(A[v_i0, v_k] - T_softmax_maxelem[v_i0])
        for i0, i1, vi in T.grid(T.int64(32), T.int64(10), T.int64(1)):
            with T.block("compute_1"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_i0, v_i1], T_softmax_maxelem[v_i0], compute_1[v_i0])
                T.writes(compute[v_i0, v_i1])
                T.block_attr({"axis": 1})
                compute[v_i0, v_i1] = A[v_i0, v_i1] - T_softmax_maxelem[v_i0] - T.log(compute_1[v_i0])

    @T.prim_func
    def matmul(A: T.Buffer((T.int64(32), T.int64(512)), "float32"), B: T.Buffer((T.int64(512), T.int64(10)), "float32"), matmul_1: T.Buffer((T.int64(32), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k, vi in T.grid(T.int64(32), T.int64(10), T.int64(512), T.int64(1)):
            with T.block("matmul"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(10), k)
                v_k = T.axis.reduce(T.int64(512), T.int64(0))
                T.reads(A[v_i0, v_k], B[v_k, v_i1])
                T.writes(matmul_1[v_i0, v_i1])
                with T.init():
                    matmul_1[v_i0, v_i1] = T.float32(0)
                matmul_1[v_i0, v_i1] = matmul_1[v_i0, v_i1] + A[v_i0, v_k] * B[v_k, v_i1]

    @T.prim_func
    def matmul1(A: T.Buffer((T.int64(32), T.int64(10)), "float32"), B: T.Buffer((T.int64(10), T.int64(512)), "float32"), matmul: T.Buffer((T.int64(32), T.int64(512)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k, vi in T.grid(T.int64(32), T.int64(512), T.int64(10), T.int64(1)):
            with T.block("matmul"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(512), k)
                v_k = T.axis.reduce(T.int64(10), T.int64(0))
                T.reads(A[v_i0, v_k], B[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + A[v_i0, v_k] * B[v_k, v_i1]

    @T.prim_func
    def matmul2(A: T.Buffer((T.int64(512), T.int64(32)), "float32"), B: T.Buffer((T.int64(32), T.int64(10)), "float32"), matmul: T.Buffer((T.int64(512), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k, vi in T.grid(T.int64(512), T.int64(10), T.int64(32), T.int64(1)):
            with T.block("matmul"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(512), i1)
                v_i1 = T.axis.spatial(T.int64(10), k)
                v_k = T.axis.reduce(T.int64(32), T.int64(0))
                T.reads(A[v_i0, v_k], B[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + A[v_i0, v_k] * B[v_k, v_i1]

    @T.prim_func
    def mean(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), T_divide: T.Buffer((T.int64(64),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red = T.alloc_buffer((T.int64(64),))
        for ax0, k0, k2, k3, vi in T.grid(T.int64(64), T.int64(32), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("A_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), k0)
                v_k0, v_k2 = T.axis.remap("RR", [k2, k3])
                v_k3 = T.axis.reduce(T.int64(32), T.int64(0))
                T.reads(A[v_k0, v_ax0, v_k2, v_k3])
                T.writes(A_red[v_ax0])
                with T.init():
                    A_red[v_ax0] = T.float32(0)
                A_red[v_ax0] = A_red[v_ax0] + A[v_k0, v_ax0, v_k2, v_k3]
        for ax0, vi in T.grid(T.int64(64), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), T.int64(0))
                T.reads(A_red[v_ax0])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = A_red[v_ax0] * T.float32(3.0517578125e-05)

    @T.prim_func
    def mean1(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), T_divide: T.Buffer((T.int64(128),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red = T.alloc_buffer((T.int64(128),))
        for ax0, k0, k2, k3, vi in T.grid(T.int64(128), T.int64(32), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("A_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), k0)
                v_k0 = T.axis.reduce(T.int64(32), k2)
                v_k2 = T.axis.reduce(T.int64(16), k3)
                v_k3 = T.axis.reduce(T.int64(16), T.int64(0))
                T.reads(A[v_k0, v_ax0, v_k2, v_k3])
                T.writes(A_red[v_ax0])
                with T.init():
                    A_red[v_ax0] = T.float32(0)
                A_red[v_ax0] = A_red[v_ax0] + A[v_k0, v_ax0, v_k2, v_k3]
        for ax0, vi in T.grid(T.int64(128), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), T.int64(0))
                T.reads(A_red[v_ax0])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = A_red[v_ax0] * T.float32(0.0001220703125)

    @T.prim_func
    def mean2(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), T_divide: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red = T.alloc_buffer((T.int64(256),))
        for ax0, k0, k2, k3, vi in T.grid(T.int64(256), T.int64(32), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("A_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), k0)
                v_k0 = T.axis.reduce(T.int64(32), k2)
                v_k2 = T.axis.reduce(T.int64(8), k3)
                v_k3 = T.axis.reduce(T.int64(8), T.int64(0))
                T.reads(A[v_k0, v_ax0, v_k2, v_k3])
                T.writes(A_red[v_ax0])
                with T.init():
                    A_red[v_ax0] = T.float32(0)
                A_red[v_ax0] = A_red[v_ax0] + A[v_k0, v_ax0, v_k2, v_k3]
        for ax0, vi in T.grid(T.int64(256), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), T.int64(0))
                T.reads(A_red[v_ax0])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = A_red[v_ax0] * T.float32(0.00048828125)

    @T.prim_func
    def mean3(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), T_divide: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red = T.alloc_buffer((T.int64(512),))
        for ax0, k0, k2, k3, vi in T.grid(T.int64(512), T.int64(32), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("A_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), k0)
                v_k0 = T.axis.reduce(T.int64(32), k2)
                v_k2 = T.axis.reduce(T.int64(4), k3)
                v_k3 = T.axis.reduce(T.int64(4), T.int64(0))
                T.reads(A[v_k0, v_ax0, v_k2, v_k3])
                T.writes(A_red[v_ax0])
                with T.init():
                    A_red[v_ax0] = T.float32(0)
                A_red[v_ax0] = A_red[v_ax0] + A[v_k0, v_ax0, v_k2, v_k3]
        for ax0, vi in T.grid(T.int64(512), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), T.int64(0))
                T.reads(A_red[v_ax0])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = A_red[v_ax0] * T.float32(0.001953125)

    @T.prim_func
    def multiply(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), B: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(32), ax3)
                v_ax3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] * B[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def multiply1(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), B: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(16), ax3)
                v_ax3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] * B[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def multiply10(A: T.Buffer((T.int64(512),), "float32"), T_multiply: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(512), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = T.float32(0.89999997615814209) * A[v_ax0]

    @T.prim_func
    def multiply11(A: T.Buffer((T.int64(512),), "float32"), T_multiply: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(512), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = T.float32(0.10000000149011612) * A[v_ax0]

    @T.prim_func
    def multiply12(A: T.Buffer((T.int64(32), T.int64(1)), "float32"), B: T.Buffer((T.int64(32), T.int64(10)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, vi in T.grid(T.int64(32), T.int64(10), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_ax0, T.int64(0)], B[v_ax0, v_ax1])
                T.writes(T_multiply[v_ax0, v_ax1])
                T_multiply[v_ax0, v_ax1] = A[v_ax0, T.int64(0)] * B[v_ax0, v_ax1]

    @T.prim_func
    def multiply13(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), B: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(4), ax3)
                v_ax3 = T.axis.spatial(T.int64(4), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] * B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply14(A: T.Buffer((T.int64(512),), "float32"), T_multiply: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(512), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = A[v_ax0] * T.float32(0.10000000149011612)

    @T.prim_func
    def multiply15(A: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(512), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.5) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply16(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(4), ax3)
                v_ax3 = T.axis.spatial(T.int64(4), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.00390625) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply17(A: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(512), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(7.62939453125e-06) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply18(A: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), B: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(4), ax3)
                v_ax3 = T.axis.spatial(T.int64(4), T.int64(0))
                T.reads(A[T.int64(0), v_ax1, T.int64(0), T.int64(0)], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), v_ax1, T.int64(0), T.int64(0)] * B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply19(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), B: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(8), ax3)
                v_ax3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] * B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply2(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), B: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(8), ax3)
                v_ax3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] * B[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def multiply20(A: T.Buffer((T.int64(256),), "float32"), T_multiply: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(256), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = A[v_ax0] * T.float32(0.10000000149011612)

    @T.prim_func
    def multiply21(A: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(256), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.5) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply22(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(8), ax3)
                v_ax3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.0009765625) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply23(A: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(256), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(4.76837158203125e-07) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply24(A: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), B: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(8), ax3)
                v_ax3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(A[T.int64(0), v_ax1, T.int64(0), T.int64(0)], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), v_ax1, T.int64(0), T.int64(0)] * B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply25(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), B: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(16), ax3)
                v_ax3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] * B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply26(A: T.Buffer((T.int64(128),), "float32"), T_multiply: T.Buffer((T.int64(128),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(128), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = A[v_ax0] * T.float32(0.10000000149011612)

    @T.prim_func
    def multiply27(A: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(128), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.5) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply28(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(16), ax3)
                v_ax3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.000244140625) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply29(A: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(128), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(2.9802322387695312e-08) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply3(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), B: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(4), ax3)
                v_ax3 = T.axis.spatial(T.int64(4), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] * B[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def multiply30(A: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32"), B: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(16), ax3)
                v_ax3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(A[T.int64(0), v_ax1, T.int64(0), T.int64(0)], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), v_ax1, T.int64(0), T.int64(0)] * B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply31(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), B: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(32), ax3)
                v_ax3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] * B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply32(A: T.Buffer((T.int64(64),), "float32"), T_multiply: T.Buffer((T.int64(64),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(64), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = A[v_ax0] * T.float32(0.10000000149011612)

    @T.prim_func
    def multiply33(A: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(64), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.5) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply34(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(32), ax3)
                v_ax3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(6.103515625e-05) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply35(A: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(64), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(1.862645149230957e-09) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply36(A: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32"), B: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(32), ax3)
                v_ax3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[T.int64(0), v_ax1, T.int64(0), T.int64(0)], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), v_ax1, T.int64(0), T.int64(0)] * B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply37(A: T.Buffer((T.int64(64), T.int64(3), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(64), T.int64(3), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(64), T.int64(3), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), ax1)
                v_ax1, v_ax2 = T.axis.remap("SS", [ax2, ax3])
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.00050000002374872565) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply38(A: T.Buffer((T.int64(64), T.int64(3), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(64), T.int64(3), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(64), T.int64(3), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), ax1)
                v_ax1, v_ax2 = T.axis.remap("SS", [ax2, ax3])
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.89999997615814209) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply39(A: T.Buffer((T.int64(64), T.int64(3), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(64), T.int64(3), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(64), T.int64(3), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), ax1)
                v_ax1, v_ax2 = T.axis.remap("SS", [ax2, ax3])
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.10000000149011612) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply4(A: T.Buffer((T.int64(64),), "float32"), T_multiply: T.Buffer((T.int64(64),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(64), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = T.float32(0.89999997615814209) * A[v_ax0]

    @T.prim_func
    def multiply40(A: T.Buffer((T.int64(64),), "float32"), T_multiply: T.Buffer((T.int64(64),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(64), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = T.float32(0.00050000002374872565) * A[v_ax0]

    @T.prim_func
    def multiply41(A: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(64), T.int64(64), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.00050000002374872565) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply42(A: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(64), T.int64(64), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.89999997615814209) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply43(A: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(64), T.int64(64), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.10000000149011612) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply44(A: T.Buffer((T.int64(128), T.int64(64), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(128), T.int64(64), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(128), T.int64(64), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.00050000002374872565) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply45(A: T.Buffer((T.int64(128), T.int64(64), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(128), T.int64(64), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(128), T.int64(64), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.89999997615814209) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply46(A: T.Buffer((T.int64(128), T.int64(64), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(128), T.int64(64), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(128), T.int64(64), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.10000000149011612) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply47(A: T.Buffer((T.int64(128),), "float32"), T_multiply: T.Buffer((T.int64(128),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(128), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = T.float32(0.00050000002374872565) * A[v_ax0]

    @T.prim_func
    def multiply48(A: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(128), T.int64(128), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.00050000002374872565) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply49(A: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(128), T.int64(128), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.89999997615814209) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply5(A: T.Buffer((T.int64(64),), "float32"), T_multiply: T.Buffer((T.int64(64),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(64), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = T.float32(0.10000000149011612) * A[v_ax0]

    @T.prim_func
    def multiply50(A: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(128), T.int64(128), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.10000000149011612) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply51(A: T.Buffer((T.int64(128), T.int64(64), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(128), T.int64(64), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(128), T.int64(64), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.00050000002374872565) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply52(A: T.Buffer((T.int64(128), T.int64(64), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(128), T.int64(64), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(128), T.int64(64), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.89999997615814209) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply53(A: T.Buffer((T.int64(128), T.int64(64), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(128), T.int64(64), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(128), T.int64(64), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.10000000149011612) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply54(A: T.Buffer((T.int64(256), T.int64(128), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(256), T.int64(128), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(256), T.int64(128), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.00050000002374872565) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply55(A: T.Buffer((T.int64(256), T.int64(128), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(256), T.int64(128), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(256), T.int64(128), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.89999997615814209) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply56(A: T.Buffer((T.int64(256), T.int64(128), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(256), T.int64(128), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(256), T.int64(128), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.10000000149011612) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply57(A: T.Buffer((T.int64(256),), "float32"), T_multiply: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(256), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = T.float32(0.00050000002374872565) * A[v_ax0]

    @T.prim_func
    def multiply58(A: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(256), T.int64(256), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.00050000002374872565) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply59(A: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(256), T.int64(256), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.89999997615814209) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply6(A: T.Buffer((T.int64(128),), "float32"), T_multiply: T.Buffer((T.int64(128),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(128), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = T.float32(0.89999997615814209) * A[v_ax0]

    @T.prim_func
    def multiply60(A: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(256), T.int64(256), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.10000000149011612) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply61(A: T.Buffer((T.int64(256), T.int64(128), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(256), T.int64(128), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(256), T.int64(128), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.00050000002374872565) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply62(A: T.Buffer((T.int64(256), T.int64(128), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(256), T.int64(128), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(256), T.int64(128), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.89999997615814209) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply63(A: T.Buffer((T.int64(256), T.int64(128), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(256), T.int64(128), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(256), T.int64(128), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.10000000149011612) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply64(A: T.Buffer((T.int64(512), T.int64(256), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(512), T.int64(256), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(512), T.int64(256), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.00050000002374872565) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply65(A: T.Buffer((T.int64(512), T.int64(256), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(512), T.int64(256), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(512), T.int64(256), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.89999997615814209) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply66(A: T.Buffer((T.int64(512), T.int64(256), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(512), T.int64(256), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(512), T.int64(256), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.10000000149011612) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply67(A: T.Buffer((T.int64(512),), "float32"), T_multiply: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(512), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = T.float32(0.00050000002374872565) * A[v_ax0]

    @T.prim_func
    def multiply68(A: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(512), T.int64(512), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.00050000002374872565) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply69(A: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(512), T.int64(512), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.89999997615814209) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply7(A: T.Buffer((T.int64(128),), "float32"), T_multiply: T.Buffer((T.int64(128),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(128), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = T.float32(0.10000000149011612) * A[v_ax0]

    @T.prim_func
    def multiply70(A: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(512), T.int64(512), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.10000000149011612) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply71(A: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(512), T.int64(256), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.00050000002374872565) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply72(A: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(512), T.int64(256), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.89999997615814209) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply73(A: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(512), T.int64(256), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.10000000149011612) * A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def multiply74(A: T.Buffer((T.int64(512), T.int64(10)), "float32"), T_multiply: T.Buffer((T.int64(512), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, vi in T.grid(T.int64(512), T.int64(10), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_ax0, v_ax1])
                T.writes(T_multiply[v_ax0, v_ax1])
                T_multiply[v_ax0, v_ax1] = T.float32(0.00050000002374872565) * A[v_ax0, v_ax1]

    @T.prim_func
    def multiply75(A: T.Buffer((T.int64(512), T.int64(10)), "float32"), T_multiply: T.Buffer((T.int64(512), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, vi in T.grid(T.int64(512), T.int64(10), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_ax0, v_ax1])
                T.writes(T_multiply[v_ax0, v_ax1])
                T_multiply[v_ax0, v_ax1] = T.float32(0.89999997615814209) * A[v_ax0, v_ax1]

    @T.prim_func
    def multiply76(A: T.Buffer((T.int64(512), T.int64(10)), "float32"), T_multiply: T.Buffer((T.int64(512), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, vi in T.grid(T.int64(512), T.int64(10), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_ax0, v_ax1])
                T.writes(T_multiply[v_ax0, v_ax1])
                T_multiply[v_ax0, v_ax1] = T.float32(0.10000000149011612) * A[v_ax0, v_ax1]

    @T.prim_func
    def multiply77(A: T.Buffer((T.int64(10),), "float32"), T_multiply: T.Buffer((T.int64(10),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(10), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = T.float32(0.00050000002374872565) * A[v_ax0]

    @T.prim_func
    def multiply78(A: T.Buffer((T.int64(10),), "float32"), T_multiply: T.Buffer((T.int64(10),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(10), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = T.float32(0.89999997615814209) * A[v_ax0]

    @T.prim_func
    def multiply79(A: T.Buffer((T.int64(10),), "float32"), T_multiply: T.Buffer((T.int64(10),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(10), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = T.float32(0.10000000149011612) * A[v_ax0]

    @T.prim_func
    def multiply8(A: T.Buffer((T.int64(256),), "float32"), T_multiply: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(256), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = T.float32(0.89999997615814209) * A[v_ax0]

    @T.prim_func
    def multiply9(A: T.Buffer((T.int64(256),), "float32"), T_multiply: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(256), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), T.int64(0))
                T.reads(A[v_ax0])
                T.writes(T_multiply[v_ax0])
                T_multiply[v_ax0] = T.float32(0.10000000149011612) * A[v_ax0]

    @T.prim_func
    def nll_loss_without_weight(A: T.Buffer((T.int64(32), T.int64(10)), "float32"), B: T.Buffer((T.int64(32),), "int64"), T_divide: T.Buffer((), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_full = T.alloc_buffer((T.int64(10),))
        nll_loss = T.alloc_buffer((T.int64(32),))
        nll_loss_red = T.alloc_buffer(())
        nll_loss_1 = T.alloc_buffer((T.int64(32),))
        nll_loss_red_1 = T.alloc_buffer(())
        for ax0, vi in T.grid(T.int64(10), T.int64(1)):
            with T.block("T_full"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads()
                T.writes(T_full[v_ax0])
                T_full[v_ax0] = T.float32(1)
        for ax0, vi in T.grid(T.int64(32), T.int64(1)):
            with T.block("nll_loss"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(B[v_ax0], A[v_ax0, B[v_ax0]], T_full[B[v_ax0]])
                T.writes(nll_loss[v_ax0])
                nll_loss[v_ax0] = T.Select(B[v_ax0] != T.int64(-100), (T.float32(0) - A[v_ax0, B[v_ax0]]) * T_full[B[v_ax0]], T.float32(0))
        for k0, vi in T.grid(T.int64(32), T.int64(1)):
            with T.block("nll_loss_red"):
                vi = T.axis.spatial(T.int64(1), k0)
                v_k0 = T.axis.reduce(T.int64(32), T.int64(0))
                T.reads(nll_loss[v_k0])
                T.writes(nll_loss_red[()])
                with T.init():
                    nll_loss_red[()] = T.float32(0)
                nll_loss_red[()] = nll_loss_red[()] + nll_loss[v_k0]
        for ax0, vi in T.grid(T.int64(32), T.int64(1)):
            with T.block("nll_loss_1"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(B[v_ax0], T_full[B[v_ax0]])
                T.writes(nll_loss_1[v_ax0])
                nll_loss_1[v_ax0] = T.Select(B[v_ax0] != T.int64(-100), T_full[B[v_ax0]], T.float32(0))
        for k0, vi in T.grid(T.int64(32), T.int64(1)):
            with T.block("nll_loss_red_1"):
                vi = T.axis.spatial(T.int64(1), k0)
                v_k0 = T.axis.reduce(T.int64(32), T.int64(0))
                T.reads(nll_loss_1[v_k0])
                T.writes(nll_loss_red_1[()])
                with T.init():
                    nll_loss_red_1[()] = T.float32(0)
                nll_loss_red_1[()] = nll_loss_red_1[()] + nll_loss_1[v_k0]
        for vi in range(T.int64(1)):
            with T.block("T_divide"):
                vi_1 = T.axis.spatial(1, T.int64(0))
                T.reads(nll_loss_red[()], nll_loss_red_1[()])
                T.writes(T_divide[()])
                T_divide[()] = nll_loss_red[()] / nll_loss_red_1[()]

    @T.prim_func
    def ones(T_full: T.Buffer((), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for vi in range(T.int64(1)):
            with T.block("T_full"):
                vi_1 = T.axis.spatial(1, T.int64(0))
                T.reads()
                T.writes(T_full[()])
                T_full[()] = T.float32(1)

    @T.prim_func
    def relu(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), compute: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(64), i2)
                v_i2 = T.axis.spatial(T.int64(32), i3)
                v_i3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                compute[v_i0, v_i1, v_i2, v_i3] = T.max(A[v_i0, v_i1, v_i2, v_i3], T.float32(0))

    @T.prim_func
    def relu1(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), compute: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(128), i2)
                v_i2 = T.axis.spatial(T.int64(16), i3)
                v_i3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                compute[v_i0, v_i1, v_i2, v_i3] = T.max(A[v_i0, v_i1, v_i2, v_i3], T.float32(0))

    @T.prim_func
    def relu2(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), compute: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(256), i2)
                v_i2 = T.axis.spatial(T.int64(8), i3)
                v_i3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                compute[v_i0, v_i1, v_i2, v_i3] = T.max(A[v_i0, v_i1, v_i2, v_i3], T.float32(0))

    @T.prim_func
    def relu3(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), compute: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(512), i2)
                v_i2 = T.axis.spatial(T.int64(4), i3)
                v_i3 = T.axis.spatial(T.int64(4), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                compute[v_i0, v_i1, v_i2, v_i3] = T.max(A[v_i0, v_i1, v_i2, v_i3], T.float32(0))

    @T.prim_func
    def reshape(A: T.Buffer((T.int64(32), T.int64(512), T.int64(1), T.int64(1)), "float32"), T_reshape: T.Buffer((T.int64(32), T.int64(512)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, vi in T.grid(T.int64(32), T.int64(512), T.int64(1)):
            with T.block("T_reshape"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), T.int64(0))
                T.reads(A[(v_ax1 // T.int64(512) + v_ax0) % T.int64(32), v_ax1 % T.int64(512), T.int64(0), T.int64(0)])
                T.writes(T_reshape[v_ax0, v_ax1])
                T_reshape[v_ax0, v_ax1] = A[(v_ax1 // T.int64(512) + v_ax0) % T.int64(32), v_ax1 % T.int64(512), T.int64(0), T.int64(0)]

    @T.prim_func
    def reshape1(A: T.Buffer((T.int64(32), T.int64(512)), "float32"), T_reshape: T.Buffer((T.int64(32), T.int64(512), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(512), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_reshape"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[((v_ax1 + v_ax2 + v_ax3) // T.int64(512) + v_ax0) % T.int64(32), (v_ax1 + v_ax2 + v_ax3) % T.int64(512)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[((v_ax1 + v_ax2 + v_ax3) // T.int64(512) + v_ax0) % T.int64(32), (v_ax1 + v_ax2 + v_ax3) % T.int64(512)]

    @T.prim_func
    def squeeze(A: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), T_squeeze: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(512), T.int64(1)):
            with T.block("T_squeeze"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), T.int64(0))
                T.reads(A[T.int64(0), v_ax0, T.int64(0), T.int64(0)])
                T.writes(T_squeeze[v_ax0])
                T_squeeze[v_ax0] = A[T.int64(0), v_ax0, T.int64(0), T.int64(0)]

    @T.prim_func
    def squeeze1(A: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), T_squeeze: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(256), T.int64(1)):
            with T.block("T_squeeze"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), T.int64(0))
                T.reads(A[T.int64(0), v_ax0, T.int64(0), T.int64(0)])
                T.writes(T_squeeze[v_ax0])
                T_squeeze[v_ax0] = A[T.int64(0), v_ax0, T.int64(0), T.int64(0)]

    @T.prim_func
    def squeeze2(A: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32"), T_squeeze: T.Buffer((T.int64(128),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(128), T.int64(1)):
            with T.block("T_squeeze"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), T.int64(0))
                T.reads(A[T.int64(0), v_ax0, T.int64(0), T.int64(0)])
                T.writes(T_squeeze[v_ax0])
                T_squeeze[v_ax0] = A[T.int64(0), v_ax0, T.int64(0), T.int64(0)]

    @T.prim_func
    def squeeze3(A: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32"), T_squeeze: T.Buffer((T.int64(64),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(64), T.int64(1)):
            with T.block("T_squeeze"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), T.int64(0))
                T.reads(A[T.int64(0), v_ax0, T.int64(0), T.int64(0)])
                T.writes(T_squeeze[v_ax0])
                T_squeeze[v_ax0] = A[T.int64(0), v_ax0, T.int64(0), T.int64(0)]

    @T.prim_func
    def strided_slice(A: T.Buffer((T.int64(512), T.int64(256), T.int64(2), T.int64(2)), "float32"), T_strided_slice_with_axes: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(512), T.int64(256), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_strided_slice_with_axes"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3])
                T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def strided_slice1(A: T.Buffer((T.int64(512), T.int64(256), T.int64(4), T.int64(4)), "float32"), T_strided_slice_with_axes: T.Buffer((T.int64(512), T.int64(256), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(512), T.int64(256), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_strided_slice_with_axes"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3])
                T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def strided_slice2(A: T.Buffer((T.int64(256), T.int64(128), T.int64(2), T.int64(2)), "float32"), T_strided_slice_with_axes: T.Buffer((T.int64(256), T.int64(128), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(256), T.int64(128), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_strided_slice_with_axes"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3])
                T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def strided_slice3(A: T.Buffer((T.int64(256), T.int64(128), T.int64(4), T.int64(4)), "float32"), T_strided_slice_with_axes: T.Buffer((T.int64(256), T.int64(128), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(256), T.int64(128), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_strided_slice_with_axes"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3])
                T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def strided_slice4(A: T.Buffer((T.int64(128), T.int64(64), T.int64(2), T.int64(2)), "float32"), T_strided_slice_with_axes: T.Buffer((T.int64(128), T.int64(64), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(128), T.int64(64), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_strided_slice_with_axes"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3])
                T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def strided_slice5(A: T.Buffer((T.int64(128), T.int64(64), T.int64(4), T.int64(4)), "float32"), T_strided_slice_with_axes: T.Buffer((T.int64(128), T.int64(64), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(128), T.int64(64), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_strided_slice_with_axes"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3])
                T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def subtract(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), B: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32"), T_subtract: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(32), ax3)
                v_ax3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - B[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def subtract1(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), B: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32"), T_subtract: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(16), ax3)
                v_ax3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - B[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def subtract10(A: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32"), B: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32"), T_subtract: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(128), T.int64(128), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def subtract11(A: T.Buffer((T.int64(128), T.int64(64), T.int64(1), T.int64(1)), "float32"), B: T.Buffer((T.int64(128), T.int64(64), T.int64(1), T.int64(1)), "float32"), T_subtract: T.Buffer((T.int64(128), T.int64(64), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(128), T.int64(64), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def subtract12(A: T.Buffer((T.int64(256), T.int64(128), T.int64(3), T.int64(3)), "float32"), B: T.Buffer((T.int64(256), T.int64(128), T.int64(3), T.int64(3)), "float32"), T_subtract: T.Buffer((T.int64(256), T.int64(128), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(256), T.int64(128), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def subtract13(A: T.Buffer((T.int64(256),), "float32"), B: T.Buffer((T.int64(256),), "float32"), T_subtract: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(256), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), T.int64(0))
                T.reads(A[v_ax0], B[v_ax0])
                T.writes(T_subtract[v_ax0])
                T_subtract[v_ax0] = A[v_ax0] - B[v_ax0]

    @T.prim_func
    def subtract14(A: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), B: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), T_subtract: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(256), T.int64(256), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def subtract15(A: T.Buffer((T.int64(256), T.int64(128), T.int64(1), T.int64(1)), "float32"), B: T.Buffer((T.int64(256), T.int64(128), T.int64(1), T.int64(1)), "float32"), T_subtract: T.Buffer((T.int64(256), T.int64(128), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(256), T.int64(128), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def subtract16(A: T.Buffer((T.int64(512), T.int64(256), T.int64(3), T.int64(3)), "float32"), B: T.Buffer((T.int64(512), T.int64(256), T.int64(3), T.int64(3)), "float32"), T_subtract: T.Buffer((T.int64(512), T.int64(256), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(512), T.int64(256), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def subtract17(A: T.Buffer((T.int64(512),), "float32"), B: T.Buffer((T.int64(512),), "float32"), T_subtract: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(512), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), T.int64(0))
                T.reads(A[v_ax0], B[v_ax0])
                T.writes(T_subtract[v_ax0])
                T_subtract[v_ax0] = A[v_ax0] - B[v_ax0]

    @T.prim_func
    def subtract18(A: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), B: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), T_subtract: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(512), T.int64(512), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def subtract19(A: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32"), B: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32"), T_subtract: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(512), T.int64(256), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def subtract2(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), B: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), T_subtract: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(8), ax3)
                v_ax3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - B[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def subtract20(A: T.Buffer((T.int64(512), T.int64(10)), "float32"), B: T.Buffer((T.int64(512), T.int64(10)), "float32"), T_subtract: T.Buffer((T.int64(512), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, vi in T.grid(T.int64(512), T.int64(10), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                T.writes(T_subtract[v_ax0, v_ax1])
                T_subtract[v_ax0, v_ax1] = A[v_ax0, v_ax1] - B[v_ax0, v_ax1]

    @T.prim_func
    def subtract21(A: T.Buffer((T.int64(10),), "float32"), B: T.Buffer((T.int64(10),), "float32"), T_subtract: T.Buffer((T.int64(10),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(10), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_ax0], B[v_ax0])
                T.writes(T_subtract[v_ax0])
                T_subtract[v_ax0] = A[v_ax0] - B[v_ax0]

    @T.prim_func
    def subtract3(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), B: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), T_subtract: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(4), ax3)
                v_ax3 = T.axis.spatial(T.int64(4), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - B[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func
    def subtract4(A: T.Buffer((T.int64(32), T.int64(10)), "float32"), B: T.Buffer((T.int64(32), T.int64(10)), "float32"), T_subtract: T.Buffer((T.int64(32), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, vi in T.grid(T.int64(32), T.int64(10), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                T.writes(T_subtract[v_ax0, v_ax1])
                T_subtract[v_ax0, v_ax1] = A[v_ax0, v_ax1] - B[v_ax0, v_ax1]

    @T.prim_func
    def subtract5(A: T.Buffer((T.int64(64), T.int64(3), T.int64(3), T.int64(3)), "float32"), B: T.Buffer((T.int64(64), T.int64(3), T.int64(3), T.int64(3)), "float32"), T_subtract: T.Buffer((T.int64(64), T.int64(3), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(64), T.int64(3), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), ax1)
                v_ax1, v_ax2 = T.axis.remap("SS", [ax2, ax3])
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def subtract6(A: T.Buffer((T.int64(64),), "float32"), B: T.Buffer((T.int64(64),), "float32"), T_subtract: T.Buffer((T.int64(64),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(64), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), T.int64(0))
                T.reads(A[v_ax0], B[v_ax0])
                T.writes(T_subtract[v_ax0])
                T_subtract[v_ax0] = A[v_ax0] - B[v_ax0]

    @T.prim_func
    def subtract7(A: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32"), B: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32"), T_subtract: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(64), T.int64(64), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def subtract8(A: T.Buffer((T.int64(128), T.int64(64), T.int64(3), T.int64(3)), "float32"), B: T.Buffer((T.int64(128), T.int64(64), T.int64(3), T.int64(3)), "float32"), T_subtract: T.Buffer((T.int64(128), T.int64(64), T.int64(3), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(128), T.int64(64), T.int64(3), T.int64(3), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(3), ax3)
                v_ax3 = T.axis.spatial(T.int64(3), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - B[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def subtract9(A: T.Buffer((T.int64(128),), "float32"), B: T.Buffer((T.int64(128),), "float32"), T_subtract: T.Buffer((T.int64(128),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(128), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), T.int64(0))
                T.reads(A[v_ax0], B[v_ax0])
                T.writes(T_subtract[v_ax0])
                T_subtract[v_ax0] = A[v_ax0] - B[v_ax0]

    @T.prim_func
    def sum(A: T.Buffer((T.int64(32), T.int64(10)), "float32"), A_red: T.Buffer((T.int64(32), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, k1, vi in T.grid(T.int64(32), T.int64(1), T.int64(10), T.int64(1)):
            with T.block("A_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(1), k1)
                v_k1 = T.axis.reduce(T.int64(10), T.int64(0))
                T.reads(A[v_ax0, v_k1])
                T.writes(A_red[v_ax0, v_ax1])
                with T.init():
                    A_red[v_ax0, v_ax1] = T.float32(0)
                A_red[v_ax0, v_ax1] = A_red[v_ax0, v_ax1] + A[v_ax0, v_k1]

    @T.prim_func
    def te_nll_loss_backward_no_weight(A: T.Buffer((), "float32"), B: T.Buffer((T.int64(32), T.int64(10)), "float32"), C: T.Buffer((T.int64(32),), "int64"), pred_grad: T.Buffer((T.int64(32), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_full = T.alloc_buffer((T.int64(10),))
        all_weights = T.alloc_buffer((T.int64(32),))
        T_broadcast_to = T.alloc_buffer((T.int64(32),))
        all_weights_red = T.alloc_buffer(())
        T_divide = T.alloc_buffer((T.int64(32),))
        for ax0, vi in T.grid(T.int64(10), T.int64(1)):
            with T.block("T_full"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads()
                T.writes(T_full[v_ax0])
                T_full[v_ax0] = T.float32(1)
        for i0, vi in T.grid(T.int64(32), T.int64(1)):
            with T.block("all_weights"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(T_full[C[v_i0]], C[v_i0])
                T.writes(all_weights[v_i0])
                all_weights[v_i0] = T_full[C[v_i0]]
        for ax0, vi in T.grid(T.int64(32), T.int64(1)):
            with T.block("T_broadcast_to"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[()])
                T.writes(T_broadcast_to[v_ax0])
                T_broadcast_to[v_ax0] = A[()]
        for k0, vi in T.grid(T.int64(32), T.int64(1)):
            with T.block("all_weights_red"):
                vi = T.axis.spatial(T.int64(1), k0)
                v_k0 = T.axis.reduce(T.int64(32), T.int64(0))
                T.reads(all_weights[v_k0])
                T.writes(all_weights_red[()])
                with T.init():
                    all_weights_red[()] = T.float32(0)
                all_weights_red[()] = all_weights_red[()] + all_weights[v_k0]
        for ax0, vi in T.grid(T.int64(32), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(T_broadcast_to[v_ax0], all_weights_red[()])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = T_broadcast_to[v_ax0] / all_weights_red[()]
        for i0, i1, vi in T.grid(T.int64(32), T.int64(10), T.int64(1)):
            with T.block("pred_grad"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(C[v_i0], all_weights[v_i0], T_divide[v_i0])
                T.writes(pred_grad[v_i0, v_i1])
                pred_grad[v_i0, v_i1] = T.Select(v_i1 == C[v_i0], all_weights[v_i0] * T.float32(-1) * T_divide[v_i0], T.float32(0))

    @T.prim_func
    def tir_exp(A: T.Buffer((T.int64(32), T.int64(10)), "float32"), compute: T.Buffer((T.int64(32), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, vi in T.grid(T.int64(32), T.int64(10), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(10), T.int64(0))
                T.reads(A[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.exp(A[v_i0, v_i1])

    @T.prim_func
    def tir_negative(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), compute: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(512), i2)
                v_i2 = T.axis.spatial(T.int64(4), i3)
                v_i3 = T.axis.spatial(T.int64(4), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                compute[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, v_i2, v_i3] * T.float32(-1)

    @T.prim_func
    def tir_negative1(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), compute: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(256), i2)
                v_i2 = T.axis.spatial(T.int64(8), i3)
                v_i3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                compute[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, v_i2, v_i3] * T.float32(-1)

    @T.prim_func
    def tir_negative2(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), compute: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(128), i2)
                v_i2 = T.axis.spatial(T.int64(16), i3)
                v_i3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                compute[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, v_i2, v_i3] * T.float32(-1)

    @T.prim_func
    def tir_negative3(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), compute: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(32), i1)
                v_i1 = T.axis.spatial(T.int64(64), i2)
                v_i2 = T.axis.spatial(T.int64(32), i3)
                v_i3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                compute[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, v_i2, v_i3] * T.float32(-1)

    @T.prim_func
    def tir_sqrt(A: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32"), compute: T.Buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, vi in T.grid(T.int64(1), T.int64(64), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(1), i1)
                v_i1 = T.axis.spatial(T.int64(64), i2)
                v_i2 = T.axis.spatial(T.int64(1), i3)
                v_i3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                compute[v_i0, v_i1, v_i2, v_i3] = T.sqrt(A[v_i0, v_i1, v_i2, v_i3])

    @T.prim_func
    def tir_sqrt1(A: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32"), compute: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, vi in T.grid(T.int64(1), T.int64(128), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(1), i1)
                v_i1 = T.axis.spatial(T.int64(128), i2)
                v_i2 = T.axis.spatial(T.int64(1), i3)
                v_i3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                compute[v_i0, v_i1, v_i2, v_i3] = T.sqrt(A[v_i0, v_i1, v_i2, v_i3])

    @T.prim_func
    def tir_sqrt2(A: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), compute: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, vi in T.grid(T.int64(1), T.int64(256), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(1), i1)
                v_i1 = T.axis.spatial(T.int64(256), i2)
                v_i2 = T.axis.spatial(T.int64(1), i3)
                v_i3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                compute[v_i0, v_i1, v_i2, v_i3] = T.sqrt(A[v_i0, v_i1, v_i2, v_i3])

    @T.prim_func
    def tir_sqrt3(A: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), compute: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, vi in T.grid(T.int64(1), T.int64(512), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("compute"):
                vi = T.axis.spatial(T.int64(1), i0)
                v_i0 = T.axis.spatial(T.int64(1), i1)
                v_i1 = T.axis.spatial(T.int64(512), i2)
                v_i2 = T.axis.spatial(T.int64(1), i3)
                v_i3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                compute[v_i0, v_i1, v_i2, v_i3] = T.sqrt(A[v_i0, v_i1, v_i2, v_i3])

    @T.prim_func
    def transpose(A: T.Buffer((T.int64(512), T.int64(10)), "float32"), T_transpose: T.Buffer((T.int64(10), T.int64(512)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, vi in T.grid(T.int64(10), T.int64(512), T.int64(1)):
            with T.block("T_transpose"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(10), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), T.int64(0))
                T.reads(A[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = A[v_ax1, v_ax0]

    @T.prim_func
    def transpose1(A: T.Buffer((T.int64(32), T.int64(512)), "float32"), T_transpose: T.Buffer((T.int64(512), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, vi in T.grid(T.int64(512), T.int64(32), T.int64(1)):
            with T.block("T_transpose"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), ax1)
                v_ax1 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = A[v_ax1, v_ax0]

    @T.prim_func
    def variance(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32"), T_divide: T.Buffer((T.int64(64),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)))
        T_divide_1 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)))
        T_subtract = T.alloc_buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)))
        T_multiply = T.alloc_buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)))
        T_multiply_red = T.alloc_buffer((T.int64(64),))
        for ax0, ax1, ax2, ax3, k0, k2, k3, vi in T.grid(T.int64(1), T.int64(64), T.int64(1), T.int64(1), T.int64(32), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("A_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), k0)
                v_k0, v_k2 = T.axis.remap("RR", [k2, k3])
                v_k3 = T.axis.reduce(T.int64(32), T.int64(0))
                T.reads(A[v_k0, v_ax1, v_k2, v_k3])
                T.writes(A_red[v_ax0, v_ax1, v_ax2, v_ax3])
                with T.init():
                    A_red[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                A_red[v_ax0, v_ax1, v_ax2, v_ax3] = A_red[v_ax0, v_ax1, v_ax2, v_ax3] + A[v_k0, v_ax1, v_k2, v_k3]
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(64), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A_red[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_divide_1[v_ax0, v_ax1, v_ax2, v_ax3])
                T_divide_1[v_ax0, v_ax1, v_ax2, v_ax3] = A_red[v_ax0, v_ax1, v_ax2, v_ax3] * T.float32(3.0517578125e-05)
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(32), ax3)
                v_ax3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], T_divide_1[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - T_divide_1[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(32), ax3)
                v_ax3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0, k0, k2, k3, vi in T.grid(T.int64(64), T.int64(32), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("T_multiply_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), k0)
                v_k0, v_k2 = T.axis.remap("RR", [k2, k3])
                v_k3 = T.axis.reduce(T.int64(32), T.int64(0))
                T.reads(T_multiply[v_k0, v_ax0, v_k2, v_k3])
                T.writes(T_multiply_red[v_ax0])
                with T.init():
                    T_multiply_red[v_ax0] = T.float32(0)
                T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply[v_k0, v_ax0, v_k2, v_k3]
        for ax0, vi in T.grid(T.int64(64), T.int64(1)):
            with T.block("T_divide_1"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), T.int64(0))
                T.reads(T_multiply_red[v_ax0])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = T_multiply_red[v_ax0] * T.float32(3.0517578125e-05)

    @T.prim_func
    def variance1(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32"), T_divide: T.Buffer((T.int64(128),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)))
        T_divide_1 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)))
        T_subtract = T.alloc_buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)))
        T_multiply = T.alloc_buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)))
        T_multiply_red = T.alloc_buffer((T.int64(128),))
        for ax0, ax1, ax2, ax3, k0, k2, k3, vi in T.grid(T.int64(1), T.int64(128), T.int64(1), T.int64(1), T.int64(32), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("A_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), k0)
                v_k0 = T.axis.reduce(T.int64(32), k2)
                v_k2 = T.axis.reduce(T.int64(16), k3)
                v_k3 = T.axis.reduce(T.int64(16), T.int64(0))
                T.reads(A[v_k0, v_ax1, v_k2, v_k3])
                T.writes(A_red[v_ax0, v_ax1, v_ax2, v_ax3])
                with T.init():
                    A_red[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                A_red[v_ax0, v_ax1, v_ax2, v_ax3] = A_red[v_ax0, v_ax1, v_ax2, v_ax3] + A[v_k0, v_ax1, v_k2, v_k3]
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(128), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A_red[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_divide_1[v_ax0, v_ax1, v_ax2, v_ax3])
                T_divide_1[v_ax0, v_ax1, v_ax2, v_ax3] = A_red[v_ax0, v_ax1, v_ax2, v_ax3] * T.float32(0.0001220703125)
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(16), ax3)
                v_ax3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], T_divide_1[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - T_divide_1[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(16), ax3)
                v_ax3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0, k0, k2, k3, vi in T.grid(T.int64(128), T.int64(32), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("T_multiply_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), k0)
                v_k0 = T.axis.reduce(T.int64(32), k2)
                v_k2 = T.axis.reduce(T.int64(16), k3)
                v_k3 = T.axis.reduce(T.int64(16), T.int64(0))
                T.reads(T_multiply[v_k0, v_ax0, v_k2, v_k3])
                T.writes(T_multiply_red[v_ax0])
                with T.init():
                    T_multiply_red[v_ax0] = T.float32(0)
                T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply[v_k0, v_ax0, v_k2, v_k3]
        for ax0, vi in T.grid(T.int64(128), T.int64(1)):
            with T.block("T_divide_1"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), T.int64(0))
                T.reads(T_multiply_red[v_ax0])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = T_multiply_red[v_ax0] * T.float32(0.0001220703125)

    @T.prim_func
    def variance2(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32"), T_divide: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_divide_1 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_subtract = T.alloc_buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)))
        T_multiply = T.alloc_buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)))
        T_multiply_red = T.alloc_buffer((T.int64(256),))
        for ax0, ax1, ax2, ax3, k0, k2, k3, vi in T.grid(T.int64(1), T.int64(256), T.int64(1), T.int64(1), T.int64(32), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("A_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), k0)
                v_k0 = T.axis.reduce(T.int64(32), k2)
                v_k2 = T.axis.reduce(T.int64(8), k3)
                v_k3 = T.axis.reduce(T.int64(8), T.int64(0))
                T.reads(A[v_k0, v_ax1, v_k2, v_k3])
                T.writes(A_red[v_ax0, v_ax1, v_ax2, v_ax3])
                with T.init():
                    A_red[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                A_red[v_ax0, v_ax1, v_ax2, v_ax3] = A_red[v_ax0, v_ax1, v_ax2, v_ax3] + A[v_k0, v_ax1, v_k2, v_k3]
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(256), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A_red[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_divide_1[v_ax0, v_ax1, v_ax2, v_ax3])
                T_divide_1[v_ax0, v_ax1, v_ax2, v_ax3] = A_red[v_ax0, v_ax1, v_ax2, v_ax3] * T.float32(0.00048828125)
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(8), ax3)
                v_ax3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], T_divide_1[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - T_divide_1[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(8), ax3)
                v_ax3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0, k0, k2, k3, vi in T.grid(T.int64(256), T.int64(32), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("T_multiply_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), k0)
                v_k0 = T.axis.reduce(T.int64(32), k2)
                v_k2 = T.axis.reduce(T.int64(8), k3)
                v_k3 = T.axis.reduce(T.int64(8), T.int64(0))
                T.reads(T_multiply[v_k0, v_ax0, v_k2, v_k3])
                T.writes(T_multiply_red[v_ax0])
                with T.init():
                    T_multiply_red[v_ax0] = T.float32(0)
                T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply[v_k0, v_ax0, v_k2, v_k3]
        for ax0, vi in T.grid(T.int64(256), T.int64(1)):
            with T.block("T_divide_1"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), T.int64(0))
                T.reads(T_multiply_red[v_ax0])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = T_multiply_red[v_ax0] * T.float32(0.00048828125)

    @T.prim_func
    def variance3(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32"), T_divide: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_divide_1 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_subtract = T.alloc_buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)))
        T_multiply = T.alloc_buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)))
        T_multiply_red = T.alloc_buffer((T.int64(512),))
        for ax0, ax1, ax2, ax3, k0, k2, k3, vi in T.grid(T.int64(1), T.int64(512), T.int64(1), T.int64(1), T.int64(32), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("A_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), k0)
                v_k0 = T.axis.reduce(T.int64(32), k2)
                v_k2 = T.axis.reduce(T.int64(4), k3)
                v_k3 = T.axis.reduce(T.int64(4), T.int64(0))
                T.reads(A[v_k0, v_ax1, v_k2, v_k3])
                T.writes(A_red[v_ax0, v_ax1, v_ax2, v_ax3])
                with T.init():
                    A_red[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                A_red[v_ax0, v_ax1, v_ax2, v_ax3] = A_red[v_ax0, v_ax1, v_ax2, v_ax3] + A[v_k0, v_ax1, v_k2, v_k3]
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(1), T.int64(512), T.int64(1), T.int64(1), T.int64(1)):
            with T.block("T_divide"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(1), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(1), ax3)
                v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(A_red[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_divide_1[v_ax0, v_ax1, v_ax2, v_ax3])
                T_divide_1[v_ax0, v_ax1, v_ax2, v_ax3] = A_red[v_ax0, v_ax1, v_ax2, v_ax3] * T.float32(0.001953125)
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("T_subtract"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(4), ax3)
                v_ax3 = T.axis.spatial(T.int64(4), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], T_divide_1[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] - T_divide_1[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("T_multiply"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(4), ax3)
                v_ax3 = T.axis.spatial(T.int64(4), T.int64(0))
                T.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0, k0, k2, k3, vi in T.grid(T.int64(512), T.int64(32), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("T_multiply_red"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), k0)
                v_k0 = T.axis.reduce(T.int64(32), k2)
                v_k2 = T.axis.reduce(T.int64(4), k3)
                v_k3 = T.axis.reduce(T.int64(4), T.int64(0))
                T.reads(T_multiply[v_k0, v_ax0, v_k2, v_k3])
                T.writes(T_multiply_red[v_ax0])
                with T.init():
                    T_multiply_red[v_ax0] = T.float32(0)
                T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply[v_k0, v_ax0, v_k2, v_k3]
        for ax0, vi in T.grid(T.int64(512), T.int64(1)):
            with T.block("T_divide_1"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), T.int64(0))
                T.reads(T_multiply_red[v_ax0])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = T_multiply_red[v_ax0] * T.float32(0.001953125)

    @T.prim_func
    def where(A: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "bool"), B: T.Buffer((), "float32"), C: T.Buffer((), "float32"), T_where: T.Buffer((T.int64(32), T.int64(512), T.int64(4), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(512), T.int64(4), T.int64(4), T.int64(1)):
            with T.block("T_where"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(512), ax2)
                v_ax2 = T.axis.spatial(T.int64(4), ax3)
                v_ax3 = T.axis.spatial(T.int64(4), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[()], C[()])
                T.writes(T_where[v_ax0, v_ax1, v_ax2, v_ax3])
                T_where[v_ax0, v_ax1, v_ax2, v_ax3] = T.Select(T.int64(0) < T.Cast("int64", A[v_ax0, v_ax1, v_ax2, v_ax3]), B[()], C[()])

    @T.prim_func
    def where1(A: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "bool"), B: T.Buffer((), "float32"), C: T.Buffer((), "float32"), T_where: T.Buffer((T.int64(32), T.int64(256), T.int64(8), T.int64(8)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(256), T.int64(8), T.int64(8), T.int64(1)):
            with T.block("T_where"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(256), ax2)
                v_ax2 = T.axis.spatial(T.int64(8), ax3)
                v_ax3 = T.axis.spatial(T.int64(8), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[()], C[()])
                T.writes(T_where[v_ax0, v_ax1, v_ax2, v_ax3])
                T_where[v_ax0, v_ax1, v_ax2, v_ax3] = T.Select(T.int64(0) < T.Cast("int64", A[v_ax0, v_ax1, v_ax2, v_ax3]), B[()], C[()])

    @T.prim_func
    def where2(A: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "bool"), B: T.Buffer((), "float32"), C: T.Buffer((), "float32"), T_where: T.Buffer((T.int64(32), T.int64(128), T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(128), T.int64(16), T.int64(16), T.int64(1)):
            with T.block("T_where"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(128), ax2)
                v_ax2 = T.axis.spatial(T.int64(16), ax3)
                v_ax3 = T.axis.spatial(T.int64(16), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[()], C[()])
                T.writes(T_where[v_ax0, v_ax1, v_ax2, v_ax3])
                T_where[v_ax0, v_ax1, v_ax2, v_ax3] = T.Select(T.int64(0) < T.Cast("int64", A[v_ax0, v_ax1, v_ax2, v_ax3]), B[()], C[()])

    @T.prim_func
    def where3(A: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "bool"), B: T.Buffer((), "float32"), C: T.Buffer((), "float32"), T_where: T.Buffer((T.int64(32), T.int64(64), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, vi in T.grid(T.int64(32), T.int64(64), T.int64(32), T.int64(32), T.int64(1)):
            with T.block("T_where"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(32), ax1)
                v_ax1 = T.axis.spatial(T.int64(64), ax2)
                v_ax2 = T.axis.spatial(T.int64(32), ax3)
                v_ax3 = T.axis.spatial(T.int64(32), T.int64(0))
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[()], C[()])
                T.writes(T_where[v_ax0, v_ax1, v_ax2, v_ax3])
                T_where[v_ax0, v_ax1, v_ax2, v_ax3] = T.Select(T.int64(0) < T.Cast("int64", A[v_ax0, v_ax1, v_ax2, v_ax3]), B[()], C[()])

    @T.prim_func
    def zeros(T_full: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(512), T.int64(1)):
            with T.block("T_full"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(512), T.int64(0))
                T.reads()
                T.writes(T_full[v_ax0])
                T_full[v_ax0] = T.float32(0)

    @T.prim_func
    def zeros1(T_full: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(256), T.int64(1)):
            with T.block("T_full"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(256), T.int64(0))
                T.reads()
                T.writes(T_full[v_ax0])
                T_full[v_ax0] = T.float32(0)

    @T.prim_func
    def zeros2(T_full: T.Buffer((T.int64(128),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(128), T.int64(1)):
            with T.block("T_full"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(128), T.int64(0))
                T.reads()
                T.writes(T_full[v_ax0])
                T_full[v_ax0] = T.float32(0)

    @T.prim_func
    def zeros3(T_full: T.Buffer((T.int64(64),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, vi in T.grid(T.int64(64), T.int64(1)):
            with T.block("T_full"):
                vi = T.axis.spatial(T.int64(1), ax0)
                v_ax0 = T.axis.spatial(T.int64(64), T.int64(0))
                T.reads()
                T.writes(T_full[v_ax0])
                T_full[v_ax0] = T.float32(0)

    @R.function
    def backbone(input: R.Tensor((32, 3, 32, 32), dtype="float32"), conv2d_weight: R.Tensor((64, 3, 3, 3), dtype="float32"), bn_gamma: R.Tensor((64,), dtype="float32"), bn_beta: R.Tensor((64,), dtype="float32"), conv2d_weight_1: R.Tensor((64, 64, 3, 3), dtype="float32"), bn_gamma_1: R.Tensor((64,), dtype="float32"), bn_beta_1: R.Tensor((64,), dtype="float32"), conv2d_weight_2: R.Tensor((64, 64, 3, 3), dtype="float32"), bn_gamma_2: R.Tensor((64,), dtype="float32"), bn_beta_2: R.Tensor((64,), dtype="float32"), conv2d_weight_3: R.Tensor((64, 64, 3, 3), dtype="float32"), bn_gamma_3: R.Tensor((64,), dtype="float32"), bn_beta_3: R.Tensor((64,), dtype="float32"), conv2d_weight_4: R.Tensor((64, 64, 3, 3), dtype="float32"), bn_gamma_4: R.Tensor((64,), dtype="float32"), bn_beta_4: R.Tensor((64,), dtype="float32"), conv2d_weight_5: R.Tensor((128, 64, 3, 3), dtype="float32"), bn_gamma_5: R.Tensor((128,), dtype="float32"), bn_beta_5: R.Tensor((128,), dtype="float32"), conv2d_weight_6: R.Tensor((128, 128, 3, 3), dtype="float32"), bn_gamma_6: R.Tensor((128,), dtype="float32"), bn_beta_6: R.Tensor((128,), dtype="float32"), conv2d_weight_7: R.Tensor((128, 64, 1, 1), dtype="float32"), bn_gamma_7: R.Tensor((128,), dtype="float32"), bn_beta_7: R.Tensor((128,), dtype="float32"), conv2d_weight_8: R.Tensor((128, 128, 3, 3), dtype="float32"), bn_gamma_8: R.Tensor((128,), dtype="float32"), bn_beta_8: R.Tensor((128,), dtype="float32"), conv2d_weight_9: R.Tensor((128, 128, 3, 3), dtype="float32"), bn_gamma_9: R.Tensor((128,), dtype="float32"), bn_beta_9: R.Tensor((128,), dtype="float32"), conv2d_weight_10: R.Tensor((256, 128, 3, 3), dtype="float32"), bn_gamma_10: R.Tensor((256,), dtype="float32"), bn_beta_10: R.Tensor((256,), dtype="float32"), conv2d_weight_11: R.Tensor((256, 256, 3, 3), dtype="float32"), bn_gamma_11: R.Tensor((256,), dtype="float32"), bn_beta_11: R.Tensor((256,), dtype="float32"), conv2d_weight_12: R.Tensor((256, 128, 1, 1), dtype="float32"), bn_gamma_12: R.Tensor((256,), dtype="float32"), bn_beta_12: R.Tensor((256,), dtype="float32"), conv2d_weight_13: R.Tensor((256, 256, 3, 3), dtype="float32"), bn_gamma_13: R.Tensor((256,), dtype="float32"), bn_beta_13: R.Tensor((256,), dtype="float32"), conv2d_weight_14: R.Tensor((256, 256, 3, 3), dtype="float32"), bn_gamma_14: R.Tensor((256,), dtype="float32"), bn_beta_14: R.Tensor((256,), dtype="float32"), conv2d_weight_15: R.Tensor((512, 256, 3, 3), dtype="float32"), bn_gamma_15: R.Tensor((512,), dtype="float32"), bn_beta_15: R.Tensor((512,), dtype="float32"), conv2d_weight_16: R.Tensor((512, 512, 3, 3), dtype="float32"), bn_gamma_16: R.Tensor((512,), dtype="float32"), bn_beta_16: R.Tensor((512,), dtype="float32"), conv2d_weight_17: R.Tensor((512, 256, 1, 1), dtype="float32"), bn_gamma_17: R.Tensor((512,), dtype="float32"), bn_beta_17: R.Tensor((512,), dtype="float32"), conv2d_weight_18: R.Tensor((512, 512, 3, 3), dtype="float32"), bn_gamma_18: R.Tensor((512,), dtype="float32"), bn_beta_18: R.Tensor((512,), dtype="float32"), conv2d_weight_19: R.Tensor((512, 512, 3, 3), dtype="float32"), bn_gamma_19: R.Tensor((512,), dtype="float32"), bn_beta_19: R.Tensor((512,), dtype="float32"), ln_weight: R.Tensor((512, 10), dtype="float32"), ln_bias: R.Tensor((10,), dtype="float32"), bn_mm: R.Tensor((64,), dtype="float32"), bn_mv: R.Tensor((64,), dtype="float32"), bn_mm_1: R.Tensor((64,), dtype="float32"), bn_mv_1: R.Tensor((64,), dtype="float32"), bn_mm_2: R.Tensor((64,), dtype="float32"), bn_mv_2: R.Tensor((64,), dtype="float32"), bn_mm_3: R.Tensor((64,), dtype="float32"), bn_mv_3: R.Tensor((64,), dtype="float32"), bn_mm_4: R.Tensor((64,), dtype="float32"), bn_mv_4: R.Tensor((64,), dtype="float32"), bn_mm_5: R.Tensor((128,), dtype="float32"), bn_mv_5: R.Tensor((128,), dtype="float32"), bn_mm_6: R.Tensor((128,), dtype="float32"), bn_mv_6: R.Tensor((128,), dtype="float32"), bn_mm_7: R.Tensor((128,), dtype="float32"), bn_mv_7: R.Tensor((128,), dtype="float32"), bn_mm_8: R.Tensor((128,), dtype="float32"), bn_mv_8: R.Tensor((128,), dtype="float32"), bn_mm_9: R.Tensor((128,), dtype="float32"), bn_mv_9: R.Tensor((128,), dtype="float32"), bn_mm_10: R.Tensor((256,), dtype="float32"), bn_mv_10: R.Tensor((256,), dtype="float32"), bn_mm_11: R.Tensor((256,), dtype="float32"), bn_mv_11: R.Tensor((256,), dtype="float32"), bn_mm_12: R.Tensor((256,), dtype="float32"), bn_mv_12: R.Tensor((256,), dtype="float32"), bn_mm_13: R.Tensor((256,), dtype="float32"), bn_mv_13: R.Tensor((256,), dtype="float32"), bn_mm_14: R.Tensor((256,), dtype="float32"), bn_mv_14: R.Tensor((256,), dtype="float32"), bn_mm_15: R.Tensor((512,), dtype="float32"), bn_mv_15: R.Tensor((512,), dtype="float32"), bn_mm_16: R.Tensor((512,), dtype="float32"), bn_mv_16: R.Tensor((512,), dtype="float32"), bn_mm_17: R.Tensor((512,), dtype="float32"), bn_mv_17: R.Tensor((512,), dtype="float32"), bn_mm_18: R.Tensor((512,), dtype="float32"), bn_mv_18: R.Tensor((512,), dtype="float32"), bn_mm_19: R.Tensor((512,), dtype="float32"), bn_mv_19: R.Tensor((512,), dtype="float32")) -> R.Tuple(R.Tensor((32, 10), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.conv2d, (input, conv2d_weight), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv_1 = R.call_tir(cls.expand_dims, (bn_mm,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv1 = R.call_tir(cls.subtract, (lv, lv_1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv2 = R.call_tir(cls.expand_dims, (bn_mv,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv3 = R.call_tir(cls.add, (lv2,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv4 = R.call_tir(cls.tir_sqrt, (lv3,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv5 = R.call_tir(cls.divide, (lv1, lv4), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv6 = R.call_tir(cls.expand_dims, (bn_gamma,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv7 = R.call_tir(cls.multiply, (lv5, lv6), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv8 = R.call_tir(cls.expand_dims, (bn_beta,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv9 = R.call_tir(cls.add1, (lv7, lv8), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv1_1: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv9, bn_mm, bn_mv
            lv2_1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv1_1[0]
            lv3_1: R.Tensor((64,), dtype="float32") = lv1_1[1]
            lv4_1: R.Tensor((64,), dtype="float32") = lv1_1[2]
            lv5_1 = R.call_tir(cls.relu, (lv2_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv6_1 = R.call_tir(cls.conv2d1, (lv5_1, conv2d_weight_1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv10 = R.call_tir(cls.expand_dims, (bn_mm_1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv11 = R.call_tir(cls.subtract, (lv6_1, lv10), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv12 = R.call_tir(cls.expand_dims, (bn_mv_1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv13 = R.call_tir(cls.add, (lv12,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv14 = R.call_tir(cls.tir_sqrt, (lv13,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv15 = R.call_tir(cls.divide, (lv11, lv14), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv16 = R.call_tir(cls.expand_dims, (bn_gamma_1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv17 = R.call_tir(cls.multiply, (lv15, lv16), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv18 = R.call_tir(cls.expand_dims, (bn_beta_1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv19 = R.call_tir(cls.add1, (lv17, lv18), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv7_1: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv19, bn_mm_1, bn_mv_1
            lv8_1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv7_1[0]
            lv9_1: R.Tensor((64,), dtype="float32") = lv7_1[1]
            lv10_1: R.Tensor((64,), dtype="float32") = lv7_1[2]
            lv11_1 = R.call_tir(cls.relu, (lv8_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv12_1 = R.call_tir(cls.conv2d1, (lv11_1, conv2d_weight_2), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv20 = R.call_tir(cls.expand_dims, (bn_mm_2,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv21 = R.call_tir(cls.subtract, (lv12_1, lv20), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv22 = R.call_tir(cls.expand_dims, (bn_mv_2,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv23 = R.call_tir(cls.add, (lv22,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv24 = R.call_tir(cls.tir_sqrt, (lv23,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv25 = R.call_tir(cls.divide, (lv21, lv24), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv26 = R.call_tir(cls.expand_dims, (bn_gamma_2,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv27 = R.call_tir(cls.multiply, (lv25, lv26), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv28 = R.call_tir(cls.expand_dims, (bn_beta_2,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv29 = R.call_tir(cls.add1, (lv27, lv28), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv13_1: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv29, bn_mm_2, bn_mv_2
            lv14_1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv13_1[0]
            lv15_1: R.Tensor((64,), dtype="float32") = lv13_1[1]
            lv16_1: R.Tensor((64,), dtype="float32") = lv13_1[2]
            lv17_1 = R.call_tir(cls.add2, (lv14_1, lv5_1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv18_1 = R.call_tir(cls.relu, (lv17_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv19_1 = R.call_tir(cls.conv2d1, (lv18_1, conv2d_weight_3), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv30 = R.call_tir(cls.expand_dims, (bn_mm_3,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv31 = R.call_tir(cls.subtract, (lv19_1, lv30), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv32 = R.call_tir(cls.expand_dims, (bn_mv_3,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv33 = R.call_tir(cls.add, (lv32,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv34 = R.call_tir(cls.tir_sqrt, (lv33,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv35 = R.call_tir(cls.divide, (lv31, lv34), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv36 = R.call_tir(cls.expand_dims, (bn_gamma_3,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv37 = R.call_tir(cls.multiply, (lv35, lv36), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv38 = R.call_tir(cls.expand_dims, (bn_beta_3,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv39 = R.call_tir(cls.add1, (lv37, lv38), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv20_1: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv39, bn_mm_3, bn_mv_3
            lv21_1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv20_1[0]
            lv22_1: R.Tensor((64,), dtype="float32") = lv20_1[1]
            lv23_1: R.Tensor((64,), dtype="float32") = lv20_1[2]
            lv24_1 = R.call_tir(cls.relu, (lv21_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv25_1 = R.call_tir(cls.conv2d1, (lv24_1, conv2d_weight_4), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv40 = R.call_tir(cls.expand_dims, (bn_mm_4,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv41 = R.call_tir(cls.subtract, (lv25_1, lv40), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv42 = R.call_tir(cls.expand_dims, (bn_mv_4,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv43 = R.call_tir(cls.add, (lv42,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv44 = R.call_tir(cls.tir_sqrt, (lv43,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv45 = R.call_tir(cls.divide, (lv41, lv44), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv46 = R.call_tir(cls.expand_dims, (bn_gamma_4,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv47 = R.call_tir(cls.multiply, (lv45, lv46), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv48 = R.call_tir(cls.expand_dims, (bn_beta_4,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv49 = R.call_tir(cls.add1, (lv47, lv48), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv26_1: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv49, bn_mm_4, bn_mv_4
            lv27_1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv26_1[0]
            lv28_1: R.Tensor((64,), dtype="float32") = lv26_1[1]
            lv29_1: R.Tensor((64,), dtype="float32") = lv26_1[2]
            lv30_1 = R.call_tir(cls.add2, (lv27_1, lv18_1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv31_1 = R.call_tir(cls.relu, (lv30_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv32_1 = R.call_tir(cls.conv2d2, (lv31_1, conv2d_weight_5), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv50 = R.call_tir(cls.expand_dims1, (bn_mm_5,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv51 = R.call_tir(cls.subtract1, (lv32_1, lv50), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv52 = R.call_tir(cls.expand_dims1, (bn_mv_5,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv53 = R.call_tir(cls.add3, (lv52,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv54 = R.call_tir(cls.tir_sqrt1, (lv53,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv55 = R.call_tir(cls.divide1, (lv51, lv54), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv56 = R.call_tir(cls.expand_dims1, (bn_gamma_5,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv57 = R.call_tir(cls.multiply1, (lv55, lv56), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv58 = R.call_tir(cls.expand_dims1, (bn_beta_5,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv59 = R.call_tir(cls.add4, (lv57, lv58), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv33_1: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv59, bn_mm_5, bn_mv_5
            lv34_1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv33_1[0]
            lv35_1: R.Tensor((128,), dtype="float32") = lv33_1[1]
            lv36_1: R.Tensor((128,), dtype="float32") = lv33_1[2]
            lv37_1 = R.call_tir(cls.relu1, (lv34_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv38_1 = R.call_tir(cls.conv2d3, (lv37_1, conv2d_weight_6), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv60 = R.call_tir(cls.expand_dims1, (bn_mm_6,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv61 = R.call_tir(cls.subtract1, (lv38_1, lv60), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv62 = R.call_tir(cls.expand_dims1, (bn_mv_6,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv63 = R.call_tir(cls.add3, (lv62,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv64 = R.call_tir(cls.tir_sqrt1, (lv63,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv65 = R.call_tir(cls.divide1, (lv61, lv64), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv66 = R.call_tir(cls.expand_dims1, (bn_gamma_6,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv67 = R.call_tir(cls.multiply1, (lv65, lv66), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv68 = R.call_tir(cls.expand_dims1, (bn_beta_6,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv69 = R.call_tir(cls.add4, (lv67, lv68), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv39_1: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv69, bn_mm_6, bn_mv_6
            lv40_1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv39_1[0]
            lv41_1: R.Tensor((128,), dtype="float32") = lv39_1[1]
            lv42_1: R.Tensor((128,), dtype="float32") = lv39_1[2]
            lv43_1 = R.call_tir(cls.conv2d4, (lv31_1, conv2d_weight_7), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv70 = R.call_tir(cls.expand_dims1, (bn_mm_7,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv71 = R.call_tir(cls.subtract1, (lv43_1, lv70), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv72 = R.call_tir(cls.expand_dims1, (bn_mv_7,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv73 = R.call_tir(cls.add3, (lv72,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv74 = R.call_tir(cls.tir_sqrt1, (lv73,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv75 = R.call_tir(cls.divide1, (lv71, lv74), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv76 = R.call_tir(cls.expand_dims1, (bn_gamma_7,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv77 = R.call_tir(cls.multiply1, (lv75, lv76), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv78 = R.call_tir(cls.expand_dims1, (bn_beta_7,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv79 = R.call_tir(cls.add4, (lv77, lv78), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv44_1: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv79, bn_mm_7, bn_mv_7
            lv45_1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv44_1[0]
            lv46_1: R.Tensor((128,), dtype="float32") = lv44_1[1]
            lv47_1: R.Tensor((128,), dtype="float32") = lv44_1[2]
            lv48_1 = R.call_tir(cls.add5, (lv40_1, lv45_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv49_1 = R.call_tir(cls.relu1, (lv48_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv50_1 = R.call_tir(cls.conv2d3, (lv49_1, conv2d_weight_8), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv80 = R.call_tir(cls.expand_dims1, (bn_mm_8,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv81 = R.call_tir(cls.subtract1, (lv50_1, lv80), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv82 = R.call_tir(cls.expand_dims1, (bn_mv_8,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv83 = R.call_tir(cls.add3, (lv82,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv84 = R.call_tir(cls.tir_sqrt1, (lv83,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv85 = R.call_tir(cls.divide1, (lv81, lv84), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv86 = R.call_tir(cls.expand_dims1, (bn_gamma_8,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv87 = R.call_tir(cls.multiply1, (lv85, lv86), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv88 = R.call_tir(cls.expand_dims1, (bn_beta_8,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv89 = R.call_tir(cls.add4, (lv87, lv88), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv51_1: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv89, bn_mm_8, bn_mv_8
            lv52_1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv51_1[0]
            lv53_1: R.Tensor((128,), dtype="float32") = lv51_1[1]
            lv54_1: R.Tensor((128,), dtype="float32") = lv51_1[2]
            lv55_1 = R.call_tir(cls.relu1, (lv52_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv56_1 = R.call_tir(cls.conv2d3, (lv55_1, conv2d_weight_9), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv90 = R.call_tir(cls.expand_dims1, (bn_mm_9,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv91 = R.call_tir(cls.subtract1, (lv56_1, lv90), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv92 = R.call_tir(cls.expand_dims1, (bn_mv_9,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv93 = R.call_tir(cls.add3, (lv92,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv94 = R.call_tir(cls.tir_sqrt1, (lv93,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv95 = R.call_tir(cls.divide1, (lv91, lv94), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv96 = R.call_tir(cls.expand_dims1, (bn_gamma_9,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv97 = R.call_tir(cls.multiply1, (lv95, lv96), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv98 = R.call_tir(cls.expand_dims1, (bn_beta_9,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv99 = R.call_tir(cls.add4, (lv97, lv98), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv57_1: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv99, bn_mm_9, bn_mv_9
            lv58_1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv57_1[0]
            lv59_1: R.Tensor((128,), dtype="float32") = lv57_1[1]
            lv60_1: R.Tensor((128,), dtype="float32") = lv57_1[2]
            lv61_1 = R.call_tir(cls.add5, (lv58_1, lv49_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv62_1 = R.call_tir(cls.relu1, (lv61_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv63_1 = R.call_tir(cls.conv2d5, (lv62_1, conv2d_weight_10), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv100 = R.call_tir(cls.expand_dims2, (bn_mm_10,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv101 = R.call_tir(cls.subtract2, (lv63_1, lv100), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv102 = R.call_tir(cls.expand_dims2, (bn_mv_10,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv103 = R.call_tir(cls.add6, (lv102,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv104 = R.call_tir(cls.tir_sqrt2, (lv103,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv105 = R.call_tir(cls.divide2, (lv101, lv104), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv106 = R.call_tir(cls.expand_dims2, (bn_gamma_10,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv107 = R.call_tir(cls.multiply2, (lv105, lv106), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv108 = R.call_tir(cls.expand_dims2, (bn_beta_10,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv109 = R.call_tir(cls.add7, (lv107, lv108), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv64_1: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv109, bn_mm_10, bn_mv_10
            lv65_1: R.Tensor((32, 256, 8, 8), dtype="float32") = lv64_1[0]
            lv66_1: R.Tensor((256,), dtype="float32") = lv64_1[1]
            lv67_1: R.Tensor((256,), dtype="float32") = lv64_1[2]
            lv68_1 = R.call_tir(cls.relu2, (lv65_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv69_1 = R.call_tir(cls.conv2d6, (lv68_1, conv2d_weight_11), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv110 = R.call_tir(cls.expand_dims2, (bn_mm_11,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv111 = R.call_tir(cls.subtract2, (lv69_1, lv110), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv112 = R.call_tir(cls.expand_dims2, (bn_mv_11,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv113 = R.call_tir(cls.add6, (lv112,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv114 = R.call_tir(cls.tir_sqrt2, (lv113,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv115 = R.call_tir(cls.divide2, (lv111, lv114), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv116 = R.call_tir(cls.expand_dims2, (bn_gamma_11,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv117 = R.call_tir(cls.multiply2, (lv115, lv116), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv118 = R.call_tir(cls.expand_dims2, (bn_beta_11,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv119 = R.call_tir(cls.add7, (lv117, lv118), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv70_1: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv119, bn_mm_11, bn_mv_11
            lv71_1: R.Tensor((32, 256, 8, 8), dtype="float32") = lv70_1[0]
            lv72_1: R.Tensor((256,), dtype="float32") = lv70_1[1]
            lv73_1: R.Tensor((256,), dtype="float32") = lv70_1[2]
            lv74_1 = R.call_tir(cls.conv2d7, (lv62_1, conv2d_weight_12), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv120 = R.call_tir(cls.expand_dims2, (bn_mm_12,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv121 = R.call_tir(cls.subtract2, (lv74_1, lv120), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv122 = R.call_tir(cls.expand_dims2, (bn_mv_12,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv123 = R.call_tir(cls.add6, (lv122,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv124 = R.call_tir(cls.tir_sqrt2, (lv123,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv125 = R.call_tir(cls.divide2, (lv121, lv124), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv126 = R.call_tir(cls.expand_dims2, (bn_gamma_12,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv127 = R.call_tir(cls.multiply2, (lv125, lv126), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv128 = R.call_tir(cls.expand_dims2, (bn_beta_12,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv129 = R.call_tir(cls.add7, (lv127, lv128), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv75_1: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv129, bn_mm_12, bn_mv_12
            lv76_1: R.Tensor((32, 256, 8, 8), dtype="float32") = lv75_1[0]
            lv77_1: R.Tensor((256,), dtype="float32") = lv75_1[1]
            lv78_1: R.Tensor((256,), dtype="float32") = lv75_1[2]
            lv79_1 = R.call_tir(cls.add8, (lv71_1, lv76_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv80_1 = R.call_tir(cls.relu2, (lv79_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv81_1 = R.call_tir(cls.conv2d6, (lv80_1, conv2d_weight_13), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv130 = R.call_tir(cls.expand_dims2, (bn_mm_13,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv131 = R.call_tir(cls.subtract2, (lv81_1, lv130), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv132 = R.call_tir(cls.expand_dims2, (bn_mv_13,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv133 = R.call_tir(cls.add6, (lv132,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv134 = R.call_tir(cls.tir_sqrt2, (lv133,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv135 = R.call_tir(cls.divide2, (lv131, lv134), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv136 = R.call_tir(cls.expand_dims2, (bn_gamma_13,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv137 = R.call_tir(cls.multiply2, (lv135, lv136), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv138 = R.call_tir(cls.expand_dims2, (bn_beta_13,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv139 = R.call_tir(cls.add7, (lv137, lv138), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv82_1: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv139, bn_mm_13, bn_mv_13
            lv83_1: R.Tensor((32, 256, 8, 8), dtype="float32") = lv82_1[0]
            lv84_1: R.Tensor((256,), dtype="float32") = lv82_1[1]
            lv85_1: R.Tensor((256,), dtype="float32") = lv82_1[2]
            lv86_1 = R.call_tir(cls.relu2, (lv83_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv87_1 = R.call_tir(cls.conv2d6, (lv86_1, conv2d_weight_14), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv140 = R.call_tir(cls.expand_dims2, (bn_mm_14,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv141 = R.call_tir(cls.subtract2, (lv87_1, lv140), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv142 = R.call_tir(cls.expand_dims2, (bn_mv_14,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv143 = R.call_tir(cls.add6, (lv142,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv144 = R.call_tir(cls.tir_sqrt2, (lv143,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv145 = R.call_tir(cls.divide2, (lv141, lv144), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv146 = R.call_tir(cls.expand_dims2, (bn_gamma_14,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv147 = R.call_tir(cls.multiply2, (lv145, lv146), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv148 = R.call_tir(cls.expand_dims2, (bn_beta_14,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv149 = R.call_tir(cls.add7, (lv147, lv148), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv88_1: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv149, bn_mm_14, bn_mv_14
            lv89_1: R.Tensor((32, 256, 8, 8), dtype="float32") = lv88_1[0]
            lv90_1: R.Tensor((256,), dtype="float32") = lv88_1[1]
            lv91_1: R.Tensor((256,), dtype="float32") = lv88_1[2]
            lv92_1 = R.call_tir(cls.add8, (lv89_1, lv80_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv93_1 = R.call_tir(cls.relu2, (lv92_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv94_1 = R.call_tir(cls.conv2d8, (lv93_1, conv2d_weight_15), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv150 = R.call_tir(cls.expand_dims3, (bn_mm_15,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv151 = R.call_tir(cls.subtract3, (lv94_1, lv150), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv152 = R.call_tir(cls.expand_dims3, (bn_mv_15,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv153 = R.call_tir(cls.add9, (lv152,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv154 = R.call_tir(cls.tir_sqrt3, (lv153,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv155 = R.call_tir(cls.divide3, (lv151, lv154), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv156 = R.call_tir(cls.expand_dims3, (bn_gamma_15,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv157 = R.call_tir(cls.multiply3, (lv155, lv156), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv158 = R.call_tir(cls.expand_dims3, (bn_beta_15,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv159 = R.call_tir(cls.add10, (lv157, lv158), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv95_1: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv159, bn_mm_15, bn_mv_15
            lv96_1: R.Tensor((32, 512, 4, 4), dtype="float32") = lv95_1[0]
            lv97_1: R.Tensor((512,), dtype="float32") = lv95_1[1]
            lv98_1: R.Tensor((512,), dtype="float32") = lv95_1[2]
            lv99_1 = R.call_tir(cls.relu3, (lv96_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv100_1 = R.call_tir(cls.conv2d9, (lv99_1, conv2d_weight_16), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv160 = R.call_tir(cls.expand_dims3, (bn_mm_16,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv161 = R.call_tir(cls.subtract3, (lv100_1, lv160), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv162 = R.call_tir(cls.expand_dims3, (bn_mv_16,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv163 = R.call_tir(cls.add9, (lv162,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv164 = R.call_tir(cls.tir_sqrt3, (lv163,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv165 = R.call_tir(cls.divide3, (lv161, lv164), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv166 = R.call_tir(cls.expand_dims3, (bn_gamma_16,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv167 = R.call_tir(cls.multiply3, (lv165, lv166), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv168 = R.call_tir(cls.expand_dims3, (bn_beta_16,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv169 = R.call_tir(cls.add10, (lv167, lv168), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv101_1: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv169, bn_mm_16, bn_mv_16
            lv102_1: R.Tensor((32, 512, 4, 4), dtype="float32") = lv101_1[0]
            lv103_1: R.Tensor((512,), dtype="float32") = lv101_1[1]
            lv104_1: R.Tensor((512,), dtype="float32") = lv101_1[2]
            lv105_1 = R.call_tir(cls.conv2d10, (lv93_1, conv2d_weight_17), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv170 = R.call_tir(cls.expand_dims3, (bn_mm_17,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv171 = R.call_tir(cls.subtract3, (lv105_1, lv170), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv172 = R.call_tir(cls.expand_dims3, (bn_mv_17,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv173 = R.call_tir(cls.add9, (lv172,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv174 = R.call_tir(cls.tir_sqrt3, (lv173,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv175 = R.call_tir(cls.divide3, (lv171, lv174), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv176 = R.call_tir(cls.expand_dims3, (bn_gamma_17,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv177 = R.call_tir(cls.multiply3, (lv175, lv176), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv178 = R.call_tir(cls.expand_dims3, (bn_beta_17,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv179 = R.call_tir(cls.add10, (lv177, lv178), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv106_1: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv179, bn_mm_17, bn_mv_17
            lv107_1: R.Tensor((32, 512, 4, 4), dtype="float32") = lv106_1[0]
            lv108_1: R.Tensor((512,), dtype="float32") = lv106_1[1]
            lv109_1: R.Tensor((512,), dtype="float32") = lv106_1[2]
            lv110_1 = R.call_tir(cls.add11, (lv102_1, lv107_1), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv111_1 = R.call_tir(cls.relu3, (lv110_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv112_1 = R.call_tir(cls.conv2d9, (lv111_1, conv2d_weight_18), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv180 = R.call_tir(cls.expand_dims3, (bn_mm_18,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv181 = R.call_tir(cls.subtract3, (lv112_1, lv180), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv182 = R.call_tir(cls.expand_dims3, (bn_mv_18,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv183 = R.call_tir(cls.add9, (lv182,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv184 = R.call_tir(cls.tir_sqrt3, (lv183,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv185 = R.call_tir(cls.divide3, (lv181, lv184), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv186 = R.call_tir(cls.expand_dims3, (bn_gamma_18,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv187 = R.call_tir(cls.multiply3, (lv185, lv186), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv188 = R.call_tir(cls.expand_dims3, (bn_beta_18,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv189 = R.call_tir(cls.add10, (lv187, lv188), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv113_1: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv189, bn_mm_18, bn_mv_18
            lv114_1: R.Tensor((32, 512, 4, 4), dtype="float32") = lv113_1[0]
            lv115_1: R.Tensor((512,), dtype="float32") = lv113_1[1]
            lv116_1: R.Tensor((512,), dtype="float32") = lv113_1[2]
            lv117_1 = R.call_tir(cls.relu3, (lv114_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv118_1 = R.call_tir(cls.conv2d9, (lv117_1, conv2d_weight_19), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv190 = R.call_tir(cls.expand_dims3, (bn_mm_19,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv191 = R.call_tir(cls.subtract3, (lv118_1, lv190), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv192 = R.call_tir(cls.expand_dims3, (bn_mv_19,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv193 = R.call_tir(cls.add9, (lv192,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv194 = R.call_tir(cls.tir_sqrt3, (lv193,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv195 = R.call_tir(cls.divide3, (lv191, lv194), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv196 = R.call_tir(cls.expand_dims3, (bn_gamma_19,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv197 = R.call_tir(cls.multiply3, (lv195, lv196), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv198 = R.call_tir(cls.expand_dims3, (bn_beta_19,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv199 = R.call_tir(cls.add10, (lv197, lv198), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv119_1: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv199, bn_mm_19, bn_mv_19
            lv120_1: R.Tensor((32, 512, 4, 4), dtype="float32") = lv119_1[0]
            lv121_1: R.Tensor((512,), dtype="float32") = lv119_1[1]
            lv122_1: R.Tensor((512,), dtype="float32") = lv119_1[2]
            lv123_1 = R.call_tir(cls.add11, (lv120_1, lv111_1), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv124_1 = R.call_tir(cls.relu3, (lv123_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv125_1 = R.call_tir(cls.avg_pool2d, (lv124_1,), out_sinfo=R.Tensor((32, 512, 1, 1), dtype="float32"))
            lv126_1 = R.call_tir(cls.reshape, (lv125_1,), out_sinfo=R.Tensor((32, 512), dtype="float32"))
            lv127_1 = R.call_tir(cls.matmul, (lv126_1, ln_weight), out_sinfo=R.Tensor((32, 10), dtype="float32"))
            lv128_1 = R.call_tir(cls.add12, (lv127_1, ln_bias), out_sinfo=R.Tensor((32, 10), dtype="float32"))
            gv: R.Tensor((32, 10), dtype="float32") = lv128_1
            gv1: R.Tensor((64,), dtype="float32") = lv3_1
            gv2: R.Tensor((64,), dtype="float32") = lv4_1
            gv3: R.Tensor((64,), dtype="float32") = lv9_1
            gv4: R.Tensor((64,), dtype="float32") = lv10_1
            gv5: R.Tensor((64,), dtype="float32") = lv15_1
            gv6: R.Tensor((64,), dtype="float32") = lv16_1
            gv7: R.Tensor((64,), dtype="float32") = lv22_1
            gv8: R.Tensor((64,), dtype="float32") = lv23_1
            gv9: R.Tensor((64,), dtype="float32") = lv28_1
            gv10: R.Tensor((64,), dtype="float32") = lv29_1
            gv11: R.Tensor((128,), dtype="float32") = lv35_1
            gv12: R.Tensor((128,), dtype="float32") = lv36_1
            gv13: R.Tensor((128,), dtype="float32") = lv41_1
            gv14: R.Tensor((128,), dtype="float32") = lv42_1
            gv15: R.Tensor((128,), dtype="float32") = lv46_1
            gv16: R.Tensor((128,), dtype="float32") = lv47_1
            gv17: R.Tensor((128,), dtype="float32") = lv53_1
            gv18: R.Tensor((128,), dtype="float32") = lv54_1
            gv19: R.Tensor((128,), dtype="float32") = lv59_1
            gv20: R.Tensor((128,), dtype="float32") = lv60_1
            gv21: R.Tensor((256,), dtype="float32") = lv66_1
            gv22: R.Tensor((256,), dtype="float32") = lv67_1
            gv23: R.Tensor((256,), dtype="float32") = lv72_1
            gv24: R.Tensor((256,), dtype="float32") = lv73_1
            gv25: R.Tensor((256,), dtype="float32") = lv77_1
            gv26: R.Tensor((256,), dtype="float32") = lv78_1
            gv27: R.Tensor((256,), dtype="float32") = lv84_1
            gv28: R.Tensor((256,), dtype="float32") = lv85_1
            gv29: R.Tensor((256,), dtype="float32") = lv90_1
            gv30: R.Tensor((256,), dtype="float32") = lv91_1
            gv31: R.Tensor((512,), dtype="float32") = lv97_1
            gv32: R.Tensor((512,), dtype="float32") = lv98_1
            gv33: R.Tensor((512,), dtype="float32") = lv103_1
            gv34: R.Tensor((512,), dtype="float32") = lv104_1
            gv35: R.Tensor((512,), dtype="float32") = lv108_1
            gv36: R.Tensor((512,), dtype="float32") = lv109_1
            gv37: R.Tensor((512,), dtype="float32") = lv115_1
            gv38: R.Tensor((512,), dtype="float32") = lv116_1
            gv39: R.Tensor((512,), dtype="float32") = lv121_1
            gv40: R.Tensor((512,), dtype="float32") = lv122_1
            R.output(gv, gv1, gv2, gv3, gv4, gv5, gv6, gv7, gv8, gv9, gv10, gv11, gv12, gv13, gv14, gv15, gv16, gv17, gv18, gv19, gv20, gv21, gv22, gv23, gv24, gv25, gv26, gv27, gv28, gv29, gv30, gv31, gv32, gv33, gv34, gv35, gv36, gv37, gv38, gv39, gv40)
        return (gv, gv1, gv2, gv3, gv4, gv5, gv6, gv7, gv8, gv9, gv10, gv11, gv12, gv13, gv14, gv15, gv16, gv17, gv18, gv19, gv20, gv21, gv22, gv23, gv24, gv25, gv26, gv27, gv28, gv29, gv30, gv31, gv32, gv33, gv34, gv35, gv36, gv37, gv38, gv39, gv40)

    @R.function
    def backbone_loss(input: R.Tensor((32, 3, 32, 32), dtype="float32"), conv2d_weight: R.Tensor((64, 3, 3, 3), dtype="float32"), bn_gamma: R.Tensor((64,), dtype="float32"), bn_beta: R.Tensor((64,), dtype="float32"), conv2d_weight_1: R.Tensor((64, 64, 3, 3), dtype="float32"), bn_gamma_1: R.Tensor((64,), dtype="float32"), bn_beta_1: R.Tensor((64,), dtype="float32"), conv2d_weight_2: R.Tensor((64, 64, 3, 3), dtype="float32"), bn_gamma_2: R.Tensor((64,), dtype="float32"), bn_beta_2: R.Tensor((64,), dtype="float32"), conv2d_weight_3: R.Tensor((64, 64, 3, 3), dtype="float32"), bn_gamma_3: R.Tensor((64,), dtype="float32"), bn_beta_3: R.Tensor((64,), dtype="float32"), conv2d_weight_4: R.Tensor((64, 64, 3, 3), dtype="float32"), bn_gamma_4: R.Tensor((64,), dtype="float32"), bn_beta_4: R.Tensor((64,), dtype="float32"), conv2d_weight_5: R.Tensor((128, 64, 3, 3), dtype="float32"), bn_gamma_5: R.Tensor((128,), dtype="float32"), bn_beta_5: R.Tensor((128,), dtype="float32"), conv2d_weight_6: R.Tensor((128, 128, 3, 3), dtype="float32"), bn_gamma_6: R.Tensor((128,), dtype="float32"), bn_beta_6: R.Tensor((128,), dtype="float32"), conv2d_weight_7: R.Tensor((128, 64, 1, 1), dtype="float32"), bn_gamma_7: R.Tensor((128,), dtype="float32"), bn_beta_7: R.Tensor((128,), dtype="float32"), conv2d_weight_8: R.Tensor((128, 128, 3, 3), dtype="float32"), bn_gamma_8: R.Tensor((128,), dtype="float32"), bn_beta_8: R.Tensor((128,), dtype="float32"), conv2d_weight_9: R.Tensor((128, 128, 3, 3), dtype="float32"), bn_gamma_9: R.Tensor((128,), dtype="float32"), bn_beta_9: R.Tensor((128,), dtype="float32"), conv2d_weight_10: R.Tensor((256, 128, 3, 3), dtype="float32"), bn_gamma_10: R.Tensor((256,), dtype="float32"), bn_beta_10: R.Tensor((256,), dtype="float32"), conv2d_weight_11: R.Tensor((256, 256, 3, 3), dtype="float32"), bn_gamma_11: R.Tensor((256,), dtype="float32"), bn_beta_11: R.Tensor((256,), dtype="float32"), conv2d_weight_12: R.Tensor((256, 128, 1, 1), dtype="float32"), bn_gamma_12: R.Tensor((256,), dtype="float32"), bn_beta_12: R.Tensor((256,), dtype="float32"), conv2d_weight_13: R.Tensor((256, 256, 3, 3), dtype="float32"), bn_gamma_13: R.Tensor((256,), dtype="float32"), bn_beta_13: R.Tensor((256,), dtype="float32"), conv2d_weight_14: R.Tensor((256, 256, 3, 3), dtype="float32"), bn_gamma_14: R.Tensor((256,), dtype="float32"), bn_beta_14: R.Tensor((256,), dtype="float32"), conv2d_weight_15: R.Tensor((512, 256, 3, 3), dtype="float32"), bn_gamma_15: R.Tensor((512,), dtype="float32"), bn_beta_15: R.Tensor((512,), dtype="float32"), conv2d_weight_16: R.Tensor((512, 512, 3, 3), dtype="float32"), bn_gamma_16: R.Tensor((512,), dtype="float32"), bn_beta_16: R.Tensor((512,), dtype="float32"), conv2d_weight_17: R.Tensor((512, 256, 1, 1), dtype="float32"), bn_gamma_17: R.Tensor((512,), dtype="float32"), bn_beta_17: R.Tensor((512,), dtype="float32"), conv2d_weight_18: R.Tensor((512, 512, 3, 3), dtype="float32"), bn_gamma_18: R.Tensor((512,), dtype="float32"), bn_beta_18: R.Tensor((512,), dtype="float32"), conv2d_weight_19: R.Tensor((512, 512, 3, 3), dtype="float32"), bn_gamma_19: R.Tensor((512,), dtype="float32"), bn_beta_19: R.Tensor((512,), dtype="float32"), ln_weight: R.Tensor((512, 10), dtype="float32"), ln_bias: R.Tensor((10,), dtype="float32"), bn_mm: R.Tensor((64,), dtype="float32"), bn_mv: R.Tensor((64,), dtype="float32"), bn_mm_1: R.Tensor((64,), dtype="float32"), bn_mv_1: R.Tensor((64,), dtype="float32"), bn_mm_2: R.Tensor((64,), dtype="float32"), bn_mv_2: R.Tensor((64,), dtype="float32"), bn_mm_3: R.Tensor((64,), dtype="float32"), bn_mv_3: R.Tensor((64,), dtype="float32"), bn_mm_4: R.Tensor((64,), dtype="float32"), bn_mv_4: R.Tensor((64,), dtype="float32"), bn_mm_5: R.Tensor((128,), dtype="float32"), bn_mv_5: R.Tensor((128,), dtype="float32"), bn_mm_6: R.Tensor((128,), dtype="float32"), bn_mv_6: R.Tensor((128,), dtype="float32"), bn_mm_7: R.Tensor((128,), dtype="float32"), bn_mv_7: R.Tensor((128,), dtype="float32"), bn_mm_8: R.Tensor((128,), dtype="float32"), bn_mv_8: R.Tensor((128,), dtype="float32"), bn_mm_9: R.Tensor((128,), dtype="float32"), bn_mv_9: R.Tensor((128,), dtype="float32"), bn_mm_10: R.Tensor((256,), dtype="float32"), bn_mv_10: R.Tensor((256,), dtype="float32"), bn_mm_11: R.Tensor((256,), dtype="float32"), bn_mv_11: R.Tensor((256,), dtype="float32"), bn_mm_12: R.Tensor((256,), dtype="float32"), bn_mv_12: R.Tensor((256,), dtype="float32"), bn_mm_13: R.Tensor((256,), dtype="float32"), bn_mv_13: R.Tensor((256,), dtype="float32"), bn_mm_14: R.Tensor((256,), dtype="float32"), bn_mv_14: R.Tensor((256,), dtype="float32"), bn_mm_15: R.Tensor((512,), dtype="float32"), bn_mv_15: R.Tensor((512,), dtype="float32"), bn_mm_16: R.Tensor((512,), dtype="float32"), bn_mv_16: R.Tensor((512,), dtype="float32"), bn_mm_17: R.Tensor((512,), dtype="float32"), bn_mv_17: R.Tensor((512,), dtype="float32"), bn_mm_18: R.Tensor((512,), dtype="float32"), bn_mv_18: R.Tensor((512,), dtype="float32"), bn_mm_19: R.Tensor((512,), dtype="float32"), bn_mv_19: R.Tensor((512,), dtype="float32"), targets: R.Tensor((32,), dtype="int64")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.conv2d, (input, conv2d_weight), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv_1 = R.call_tir(cls.mean, (lv,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv1 = R.call_tir(cls.expand_dims, (lv_1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv2 = R.call_tir(cls.subtract, (lv, lv1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv3 = R.call_tir(cls.variance, (lv,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv4 = R.call_tir(cls.expand_dims, (lv3,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv5 = R.call_tir(cls.add, (lv4,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv6 = R.call_tir(cls.tir_sqrt, (lv5,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv7 = R.call_tir(cls.divide, (lv2, lv6), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv8 = R.call_tir(cls.expand_dims, (bn_gamma,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv9 = R.call_tir(cls.multiply, (lv7, lv8), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv10 = R.call_tir(cls.expand_dims, (bn_beta,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv11 = R.call_tir(cls.add1, (lv9, lv10), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv12 = R.call_tir(cls.multiply4, (bn_mm,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv13 = R.call_tir(cls.multiply5, (lv_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv14 = R.call_tir(cls.add13, (lv12, lv13), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv15 = R.call_tir(cls.multiply4, (bn_mv,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv16 = R.call_tir(cls.multiply5, (lv3,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv17 = R.call_tir(cls.add13, (lv15, lv16), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv1_1: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv11, lv14, lv17
            lv2_1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv1_1[0]
            lv3_1: R.Tensor((64,), dtype="float32") = lv1_1[1]
            lv4_1: R.Tensor((64,), dtype="float32") = lv1_1[2]
            lv5_1 = R.call_tir(cls.relu, (lv2_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv6_1 = R.call_tir(cls.conv2d1, (lv5_1, conv2d_weight_1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv18 = R.call_tir(cls.mean, (lv6_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv19 = R.call_tir(cls.expand_dims, (lv18,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv20 = R.call_tir(cls.subtract, (lv6_1, lv19), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv21 = R.call_tir(cls.variance, (lv6_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv22 = R.call_tir(cls.expand_dims, (lv21,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv23 = R.call_tir(cls.add, (lv22,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv24 = R.call_tir(cls.tir_sqrt, (lv23,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv25 = R.call_tir(cls.divide, (lv20, lv24), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv26 = R.call_tir(cls.expand_dims, (bn_gamma_1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv27 = R.call_tir(cls.multiply, (lv25, lv26), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv28 = R.call_tir(cls.expand_dims, (bn_beta_1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv29 = R.call_tir(cls.add1, (lv27, lv28), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv30 = R.call_tir(cls.multiply4, (bn_mm_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv31 = R.call_tir(cls.multiply5, (lv18,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv32 = R.call_tir(cls.add13, (lv30, lv31), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv33 = R.call_tir(cls.multiply4, (bn_mv_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv34 = R.call_tir(cls.multiply5, (lv21,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv35 = R.call_tir(cls.add13, (lv33, lv34), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv7_1: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv29, lv32, lv35
            lv8_1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv7_1[0]
            lv9_1: R.Tensor((64,), dtype="float32") = lv7_1[1]
            lv10_1: R.Tensor((64,), dtype="float32") = lv7_1[2]
            lv11_1 = R.call_tir(cls.relu, (lv8_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv12_1 = R.call_tir(cls.conv2d1, (lv11_1, conv2d_weight_2), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv36 = R.call_tir(cls.mean, (lv12_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv37 = R.call_tir(cls.expand_dims, (lv36,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv38 = R.call_tir(cls.subtract, (lv12_1, lv37), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv39 = R.call_tir(cls.variance, (lv12_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv40 = R.call_tir(cls.expand_dims, (lv39,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv41 = R.call_tir(cls.add, (lv40,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv42 = R.call_tir(cls.tir_sqrt, (lv41,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv43 = R.call_tir(cls.divide, (lv38, lv42), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv44 = R.call_tir(cls.expand_dims, (bn_gamma_2,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv45 = R.call_tir(cls.multiply, (lv43, lv44), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv46 = R.call_tir(cls.expand_dims, (bn_beta_2,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv47 = R.call_tir(cls.add1, (lv45, lv46), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv48 = R.call_tir(cls.multiply4, (bn_mm_2,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv49 = R.call_tir(cls.multiply5, (lv36,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv50 = R.call_tir(cls.add13, (lv48, lv49), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv51 = R.call_tir(cls.multiply4, (bn_mv_2,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv52 = R.call_tir(cls.multiply5, (lv39,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv53 = R.call_tir(cls.add13, (lv51, lv52), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv13_1: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv47, lv50, lv53
            lv14_1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv13_1[0]
            lv15_1: R.Tensor((64,), dtype="float32") = lv13_1[1]
            lv16_1: R.Tensor((64,), dtype="float32") = lv13_1[2]
            lv17_1 = R.call_tir(cls.add2, (lv14_1, lv5_1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv18_1 = R.call_tir(cls.relu, (lv17_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv19_1 = R.call_tir(cls.conv2d1, (lv18_1, conv2d_weight_3), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv54 = R.call_tir(cls.mean, (lv19_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv55 = R.call_tir(cls.expand_dims, (lv54,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv56 = R.call_tir(cls.subtract, (lv19_1, lv55), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv57 = R.call_tir(cls.variance, (lv19_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv58 = R.call_tir(cls.expand_dims, (lv57,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv59 = R.call_tir(cls.add, (lv58,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv60 = R.call_tir(cls.tir_sqrt, (lv59,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv61 = R.call_tir(cls.divide, (lv56, lv60), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv62 = R.call_tir(cls.expand_dims, (bn_gamma_3,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv63 = R.call_tir(cls.multiply, (lv61, lv62), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv64 = R.call_tir(cls.expand_dims, (bn_beta_3,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv65 = R.call_tir(cls.add1, (lv63, lv64), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv66 = R.call_tir(cls.multiply4, (bn_mm_3,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv67 = R.call_tir(cls.multiply5, (lv54,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv68 = R.call_tir(cls.add13, (lv66, lv67), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv69 = R.call_tir(cls.multiply4, (bn_mv_3,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv70 = R.call_tir(cls.multiply5, (lv57,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv71 = R.call_tir(cls.add13, (lv69, lv70), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv20_1: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv65, lv68, lv71
            lv21_1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv20_1[0]
            lv22_1: R.Tensor((64,), dtype="float32") = lv20_1[1]
            lv23_1: R.Tensor((64,), dtype="float32") = lv20_1[2]
            lv24_1 = R.call_tir(cls.relu, (lv21_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv25_1 = R.call_tir(cls.conv2d1, (lv24_1, conv2d_weight_4), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv72 = R.call_tir(cls.mean, (lv25_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv73 = R.call_tir(cls.expand_dims, (lv72,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv74 = R.call_tir(cls.subtract, (lv25_1, lv73), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv75 = R.call_tir(cls.variance, (lv25_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv76 = R.call_tir(cls.expand_dims, (lv75,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv77 = R.call_tir(cls.add, (lv76,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv78 = R.call_tir(cls.tir_sqrt, (lv77,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv79 = R.call_tir(cls.divide, (lv74, lv78), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv80 = R.call_tir(cls.expand_dims, (bn_gamma_4,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv81 = R.call_tir(cls.multiply, (lv79, lv80), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv82 = R.call_tir(cls.expand_dims, (bn_beta_4,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv83 = R.call_tir(cls.add1, (lv81, lv82), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv84 = R.call_tir(cls.multiply4, (bn_mm_4,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv85 = R.call_tir(cls.multiply5, (lv72,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv86 = R.call_tir(cls.add13, (lv84, lv85), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv87 = R.call_tir(cls.multiply4, (bn_mv_4,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv88 = R.call_tir(cls.multiply5, (lv75,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv89 = R.call_tir(cls.add13, (lv87, lv88), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv26_1: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv83, lv86, lv89
            lv27_1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv26_1[0]
            lv28_1: R.Tensor((64,), dtype="float32") = lv26_1[1]
            lv29_1: R.Tensor((64,), dtype="float32") = lv26_1[2]
            lv30_1 = R.call_tir(cls.add2, (lv27_1, lv18_1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv31_1 = R.call_tir(cls.relu, (lv30_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv32_1 = R.call_tir(cls.conv2d2, (lv31_1, conv2d_weight_5), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv90 = R.call_tir(cls.mean1, (lv32_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv91 = R.call_tir(cls.expand_dims1, (lv90,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv92 = R.call_tir(cls.subtract1, (lv32_1, lv91), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv93 = R.call_tir(cls.variance1, (lv32_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv94 = R.call_tir(cls.expand_dims1, (lv93,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv95 = R.call_tir(cls.add3, (lv94,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv96 = R.call_tir(cls.tir_sqrt1, (lv95,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv97 = R.call_tir(cls.divide1, (lv92, lv96), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv98 = R.call_tir(cls.expand_dims1, (bn_gamma_5,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv99 = R.call_tir(cls.multiply1, (lv97, lv98), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv100 = R.call_tir(cls.expand_dims1, (bn_beta_5,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv101 = R.call_tir(cls.add4, (lv99, lv100), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv102 = R.call_tir(cls.multiply6, (bn_mm_5,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv103 = R.call_tir(cls.multiply7, (lv90,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv104 = R.call_tir(cls.add14, (lv102, lv103), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv105 = R.call_tir(cls.multiply6, (bn_mv_5,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv106 = R.call_tir(cls.multiply7, (lv93,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv107 = R.call_tir(cls.add14, (lv105, lv106), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv33_1: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv101, lv104, lv107
            lv34_1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv33_1[0]
            lv35_1: R.Tensor((128,), dtype="float32") = lv33_1[1]
            lv36_1: R.Tensor((128,), dtype="float32") = lv33_1[2]
            lv37_1 = R.call_tir(cls.relu1, (lv34_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv38_1 = R.call_tir(cls.conv2d3, (lv37_1, conv2d_weight_6), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv108 = R.call_tir(cls.mean1, (lv38_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv109 = R.call_tir(cls.expand_dims1, (lv108,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv110 = R.call_tir(cls.subtract1, (lv38_1, lv109), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv111 = R.call_tir(cls.variance1, (lv38_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv112 = R.call_tir(cls.expand_dims1, (lv111,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv113 = R.call_tir(cls.add3, (lv112,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv114 = R.call_tir(cls.tir_sqrt1, (lv113,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv115 = R.call_tir(cls.divide1, (lv110, lv114), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv116 = R.call_tir(cls.expand_dims1, (bn_gamma_6,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv117 = R.call_tir(cls.multiply1, (lv115, lv116), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv118 = R.call_tir(cls.expand_dims1, (bn_beta_6,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv119 = R.call_tir(cls.add4, (lv117, lv118), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv120 = R.call_tir(cls.multiply6, (bn_mm_6,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv121 = R.call_tir(cls.multiply7, (lv108,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv122 = R.call_tir(cls.add14, (lv120, lv121), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv123 = R.call_tir(cls.multiply6, (bn_mv_6,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv124 = R.call_tir(cls.multiply7, (lv111,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv125 = R.call_tir(cls.add14, (lv123, lv124), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv39_1: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv119, lv122, lv125
            lv40_1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv39_1[0]
            lv41_1: R.Tensor((128,), dtype="float32") = lv39_1[1]
            lv42_1: R.Tensor((128,), dtype="float32") = lv39_1[2]
            lv43_1 = R.call_tir(cls.conv2d4, (lv31_1, conv2d_weight_7), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv126 = R.call_tir(cls.mean1, (lv43_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv127 = R.call_tir(cls.expand_dims1, (lv126,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv128 = R.call_tir(cls.subtract1, (lv43_1, lv127), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv129 = R.call_tir(cls.variance1, (lv43_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv130 = R.call_tir(cls.expand_dims1, (lv129,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv131 = R.call_tir(cls.add3, (lv130,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv132 = R.call_tir(cls.tir_sqrt1, (lv131,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv133 = R.call_tir(cls.divide1, (lv128, lv132), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv134 = R.call_tir(cls.expand_dims1, (bn_gamma_7,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv135 = R.call_tir(cls.multiply1, (lv133, lv134), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv136 = R.call_tir(cls.expand_dims1, (bn_beta_7,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv137 = R.call_tir(cls.add4, (lv135, lv136), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv138 = R.call_tir(cls.multiply6, (bn_mm_7,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv139 = R.call_tir(cls.multiply7, (lv126,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv140 = R.call_tir(cls.add14, (lv138, lv139), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv141 = R.call_tir(cls.multiply6, (bn_mv_7,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv142 = R.call_tir(cls.multiply7, (lv129,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv143 = R.call_tir(cls.add14, (lv141, lv142), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv44_1: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv137, lv140, lv143
            lv45_1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv44_1[0]
            lv46_1: R.Tensor((128,), dtype="float32") = lv44_1[1]
            lv47_1: R.Tensor((128,), dtype="float32") = lv44_1[2]
            lv48_1 = R.call_tir(cls.add5, (lv40_1, lv45_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv49_1 = R.call_tir(cls.relu1, (lv48_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv50_1 = R.call_tir(cls.conv2d3, (lv49_1, conv2d_weight_8), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv144 = R.call_tir(cls.mean1, (lv50_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv145 = R.call_tir(cls.expand_dims1, (lv144,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv146 = R.call_tir(cls.subtract1, (lv50_1, lv145), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv147 = R.call_tir(cls.variance1, (lv50_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv148 = R.call_tir(cls.expand_dims1, (lv147,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv149 = R.call_tir(cls.add3, (lv148,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv150 = R.call_tir(cls.tir_sqrt1, (lv149,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv151 = R.call_tir(cls.divide1, (lv146, lv150), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv152 = R.call_tir(cls.expand_dims1, (bn_gamma_8,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv153 = R.call_tir(cls.multiply1, (lv151, lv152), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv154 = R.call_tir(cls.expand_dims1, (bn_beta_8,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv155 = R.call_tir(cls.add4, (lv153, lv154), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv156 = R.call_tir(cls.multiply6, (bn_mm_8,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv157 = R.call_tir(cls.multiply7, (lv144,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv158 = R.call_tir(cls.add14, (lv156, lv157), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv159 = R.call_tir(cls.multiply6, (bn_mv_8,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv160 = R.call_tir(cls.multiply7, (lv147,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv161 = R.call_tir(cls.add14, (lv159, lv160), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv51_1: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv155, lv158, lv161
            lv52_1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv51_1[0]
            lv53_1: R.Tensor((128,), dtype="float32") = lv51_1[1]
            lv54_1: R.Tensor((128,), dtype="float32") = lv51_1[2]
            lv55_1 = R.call_tir(cls.relu1, (lv52_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv56_1 = R.call_tir(cls.conv2d3, (lv55_1, conv2d_weight_9), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv162 = R.call_tir(cls.mean1, (lv56_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv163 = R.call_tir(cls.expand_dims1, (lv162,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv164 = R.call_tir(cls.subtract1, (lv56_1, lv163), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv165 = R.call_tir(cls.variance1, (lv56_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv166 = R.call_tir(cls.expand_dims1, (lv165,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv167 = R.call_tir(cls.add3, (lv166,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv168 = R.call_tir(cls.tir_sqrt1, (lv167,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv169 = R.call_tir(cls.divide1, (lv164, lv168), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv170 = R.call_tir(cls.expand_dims1, (bn_gamma_9,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv171 = R.call_tir(cls.multiply1, (lv169, lv170), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv172 = R.call_tir(cls.expand_dims1, (bn_beta_9,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv173 = R.call_tir(cls.add4, (lv171, lv172), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv174 = R.call_tir(cls.multiply6, (bn_mm_9,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv175 = R.call_tir(cls.multiply7, (lv162,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv176 = R.call_tir(cls.add14, (lv174, lv175), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv177 = R.call_tir(cls.multiply6, (bn_mv_9,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv178 = R.call_tir(cls.multiply7, (lv165,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv179 = R.call_tir(cls.add14, (lv177, lv178), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv57_1: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv173, lv176, lv179
            lv58_1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv57_1[0]
            lv59_1: R.Tensor((128,), dtype="float32") = lv57_1[1]
            lv60_1: R.Tensor((128,), dtype="float32") = lv57_1[2]
            lv61_1 = R.call_tir(cls.add5, (lv58_1, lv49_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv62_1 = R.call_tir(cls.relu1, (lv61_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv63_1 = R.call_tir(cls.conv2d5, (lv62_1, conv2d_weight_10), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv180 = R.call_tir(cls.mean2, (lv63_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv181 = R.call_tir(cls.expand_dims2, (lv180,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv182 = R.call_tir(cls.subtract2, (lv63_1, lv181), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv183 = R.call_tir(cls.variance2, (lv63_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv184 = R.call_tir(cls.expand_dims2, (lv183,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv185 = R.call_tir(cls.add6, (lv184,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv186 = R.call_tir(cls.tir_sqrt2, (lv185,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv187 = R.call_tir(cls.divide2, (lv182, lv186), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv188 = R.call_tir(cls.expand_dims2, (bn_gamma_10,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv189 = R.call_tir(cls.multiply2, (lv187, lv188), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv190 = R.call_tir(cls.expand_dims2, (bn_beta_10,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv191 = R.call_tir(cls.add7, (lv189, lv190), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv192 = R.call_tir(cls.multiply8, (bn_mm_10,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv193 = R.call_tir(cls.multiply9, (lv180,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv194 = R.call_tir(cls.add15, (lv192, lv193), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv195 = R.call_tir(cls.multiply8, (bn_mv_10,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv196 = R.call_tir(cls.multiply9, (lv183,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv197 = R.call_tir(cls.add15, (lv195, lv196), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv64_1: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv191, lv194, lv197
            lv65_1: R.Tensor((32, 256, 8, 8), dtype="float32") = lv64_1[0]
            lv66_1: R.Tensor((256,), dtype="float32") = lv64_1[1]
            lv67_1: R.Tensor((256,), dtype="float32") = lv64_1[2]
            lv68_1 = R.call_tir(cls.relu2, (lv65_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv69_1 = R.call_tir(cls.conv2d6, (lv68_1, conv2d_weight_11), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv198 = R.call_tir(cls.mean2, (lv69_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv199 = R.call_tir(cls.expand_dims2, (lv198,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv200 = R.call_tir(cls.subtract2, (lv69_1, lv199), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv201 = R.call_tir(cls.variance2, (lv69_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv202 = R.call_tir(cls.expand_dims2, (lv201,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv203 = R.call_tir(cls.add6, (lv202,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv204 = R.call_tir(cls.tir_sqrt2, (lv203,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv205 = R.call_tir(cls.divide2, (lv200, lv204), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv206 = R.call_tir(cls.expand_dims2, (bn_gamma_11,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv207 = R.call_tir(cls.multiply2, (lv205, lv206), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv208 = R.call_tir(cls.expand_dims2, (bn_beta_11,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv209 = R.call_tir(cls.add7, (lv207, lv208), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv210 = R.call_tir(cls.multiply8, (bn_mm_11,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv211 = R.call_tir(cls.multiply9, (lv198,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv212 = R.call_tir(cls.add15, (lv210, lv211), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv213 = R.call_tir(cls.multiply8, (bn_mv_11,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv214 = R.call_tir(cls.multiply9, (lv201,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv215 = R.call_tir(cls.add15, (lv213, lv214), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv70_1: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv209, lv212, lv215
            lv71_1: R.Tensor((32, 256, 8, 8), dtype="float32") = lv70_1[0]
            lv72_1: R.Tensor((256,), dtype="float32") = lv70_1[1]
            lv73_1: R.Tensor((256,), dtype="float32") = lv70_1[2]
            lv74_1 = R.call_tir(cls.conv2d7, (lv62_1, conv2d_weight_12), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv216 = R.call_tir(cls.mean2, (lv74_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv217 = R.call_tir(cls.expand_dims2, (lv216,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv218 = R.call_tir(cls.subtract2, (lv74_1, lv217), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv219 = R.call_tir(cls.variance2, (lv74_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv220 = R.call_tir(cls.expand_dims2, (lv219,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv221 = R.call_tir(cls.add6, (lv220,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv222 = R.call_tir(cls.tir_sqrt2, (lv221,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv223 = R.call_tir(cls.divide2, (lv218, lv222), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv224 = R.call_tir(cls.expand_dims2, (bn_gamma_12,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv225 = R.call_tir(cls.multiply2, (lv223, lv224), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv226 = R.call_tir(cls.expand_dims2, (bn_beta_12,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv227 = R.call_tir(cls.add7, (lv225, lv226), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv228 = R.call_tir(cls.multiply8, (bn_mm_12,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv229 = R.call_tir(cls.multiply9, (lv216,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv230 = R.call_tir(cls.add15, (lv228, lv229), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv231 = R.call_tir(cls.multiply8, (bn_mv_12,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv232 = R.call_tir(cls.multiply9, (lv219,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv233 = R.call_tir(cls.add15, (lv231, lv232), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv75_1: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv227, lv230, lv233
            lv76_1: R.Tensor((32, 256, 8, 8), dtype="float32") = lv75_1[0]
            lv77_1: R.Tensor((256,), dtype="float32") = lv75_1[1]
            lv78_1: R.Tensor((256,), dtype="float32") = lv75_1[2]
            lv79_1 = R.call_tir(cls.add8, (lv71_1, lv76_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv80_1 = R.call_tir(cls.relu2, (lv79_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv81_1 = R.call_tir(cls.conv2d6, (lv80_1, conv2d_weight_13), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv234 = R.call_tir(cls.mean2, (lv81_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv235 = R.call_tir(cls.expand_dims2, (lv234,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv236 = R.call_tir(cls.subtract2, (lv81_1, lv235), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv237 = R.call_tir(cls.variance2, (lv81_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv238 = R.call_tir(cls.expand_dims2, (lv237,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv239 = R.call_tir(cls.add6, (lv238,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv240 = R.call_tir(cls.tir_sqrt2, (lv239,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv241 = R.call_tir(cls.divide2, (lv236, lv240), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv242 = R.call_tir(cls.expand_dims2, (bn_gamma_13,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv243 = R.call_tir(cls.multiply2, (lv241, lv242), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv244 = R.call_tir(cls.expand_dims2, (bn_beta_13,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv245 = R.call_tir(cls.add7, (lv243, lv244), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv246 = R.call_tir(cls.multiply8, (bn_mm_13,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv247 = R.call_tir(cls.multiply9, (lv234,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv248 = R.call_tir(cls.add15, (lv246, lv247), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv249 = R.call_tir(cls.multiply8, (bn_mv_13,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv250 = R.call_tir(cls.multiply9, (lv237,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv251 = R.call_tir(cls.add15, (lv249, lv250), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv82_1: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv245, lv248, lv251
            lv83_1: R.Tensor((32, 256, 8, 8), dtype="float32") = lv82_1[0]
            lv84_1: R.Tensor((256,), dtype="float32") = lv82_1[1]
            lv85_1: R.Tensor((256,), dtype="float32") = lv82_1[2]
            lv86_1 = R.call_tir(cls.relu2, (lv83_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv87_1 = R.call_tir(cls.conv2d6, (lv86_1, conv2d_weight_14), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv252 = R.call_tir(cls.mean2, (lv87_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv253 = R.call_tir(cls.expand_dims2, (lv252,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv254 = R.call_tir(cls.subtract2, (lv87_1, lv253), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv255 = R.call_tir(cls.variance2, (lv87_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv256 = R.call_tir(cls.expand_dims2, (lv255,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv257 = R.call_tir(cls.add6, (lv256,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv258 = R.call_tir(cls.tir_sqrt2, (lv257,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv259 = R.call_tir(cls.divide2, (lv254, lv258), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv260 = R.call_tir(cls.expand_dims2, (bn_gamma_14,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv261 = R.call_tir(cls.multiply2, (lv259, lv260), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv262 = R.call_tir(cls.expand_dims2, (bn_beta_14,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv263 = R.call_tir(cls.add7, (lv261, lv262), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv264 = R.call_tir(cls.multiply8, (bn_mm_14,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv265 = R.call_tir(cls.multiply9, (lv252,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv266 = R.call_tir(cls.add15, (lv264, lv265), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv267 = R.call_tir(cls.multiply8, (bn_mv_14,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv268 = R.call_tir(cls.multiply9, (lv255,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv269 = R.call_tir(cls.add15, (lv267, lv268), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv88_1: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv263, lv266, lv269
            lv89_1: R.Tensor((32, 256, 8, 8), dtype="float32") = lv88_1[0]
            lv90_1: R.Tensor((256,), dtype="float32") = lv88_1[1]
            lv91_1: R.Tensor((256,), dtype="float32") = lv88_1[2]
            lv92_1 = R.call_tir(cls.add8, (lv89_1, lv80_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv93_1 = R.call_tir(cls.relu2, (lv92_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv94_1 = R.call_tir(cls.conv2d8, (lv93_1, conv2d_weight_15), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv270 = R.call_tir(cls.mean3, (lv94_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv271 = R.call_tir(cls.expand_dims3, (lv270,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv272 = R.call_tir(cls.subtract3, (lv94_1, lv271), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv273 = R.call_tir(cls.variance3, (lv94_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv274 = R.call_tir(cls.expand_dims3, (lv273,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv275 = R.call_tir(cls.add9, (lv274,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv276 = R.call_tir(cls.tir_sqrt3, (lv275,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv277 = R.call_tir(cls.divide3, (lv272, lv276), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv278 = R.call_tir(cls.expand_dims3, (bn_gamma_15,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv279 = R.call_tir(cls.multiply3, (lv277, lv278), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv280 = R.call_tir(cls.expand_dims3, (bn_beta_15,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv281 = R.call_tir(cls.add10, (lv279, lv280), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv282 = R.call_tir(cls.multiply10, (bn_mm_15,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv283 = R.call_tir(cls.multiply11, (lv270,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv284 = R.call_tir(cls.add16, (lv282, lv283), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv285 = R.call_tir(cls.multiply10, (bn_mv_15,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv286 = R.call_tir(cls.multiply11, (lv273,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv287 = R.call_tir(cls.add16, (lv285, lv286), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv95_1: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv281, lv284, lv287
            lv96_1: R.Tensor((32, 512, 4, 4), dtype="float32") = lv95_1[0]
            lv97_1: R.Tensor((512,), dtype="float32") = lv95_1[1]
            lv98_1: R.Tensor((512,), dtype="float32") = lv95_1[2]
            lv99_1 = R.call_tir(cls.relu3, (lv96_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv100_1 = R.call_tir(cls.conv2d9, (lv99_1, conv2d_weight_16), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv288 = R.call_tir(cls.mean3, (lv100_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv289 = R.call_tir(cls.expand_dims3, (lv288,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv290 = R.call_tir(cls.subtract3, (lv100_1, lv289), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv291 = R.call_tir(cls.variance3, (lv100_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv292 = R.call_tir(cls.expand_dims3, (lv291,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv293 = R.call_tir(cls.add9, (lv292,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv294 = R.call_tir(cls.tir_sqrt3, (lv293,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv295 = R.call_tir(cls.divide3, (lv290, lv294), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv296 = R.call_tir(cls.expand_dims3, (bn_gamma_16,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv297 = R.call_tir(cls.multiply3, (lv295, lv296), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv298 = R.call_tir(cls.expand_dims3, (bn_beta_16,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv299 = R.call_tir(cls.add10, (lv297, lv298), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv300 = R.call_tir(cls.multiply10, (bn_mm_16,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv301 = R.call_tir(cls.multiply11, (lv288,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv302 = R.call_tir(cls.add16, (lv300, lv301), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv303 = R.call_tir(cls.multiply10, (bn_mv_16,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv304 = R.call_tir(cls.multiply11, (lv291,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv305 = R.call_tir(cls.add16, (lv303, lv304), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv101_1: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv299, lv302, lv305
            lv102_1: R.Tensor((32, 512, 4, 4), dtype="float32") = lv101_1[0]
            lv103_1: R.Tensor((512,), dtype="float32") = lv101_1[1]
            lv104_1: R.Tensor((512,), dtype="float32") = lv101_1[2]
            lv105_1 = R.call_tir(cls.conv2d10, (lv93_1, conv2d_weight_17), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv306 = R.call_tir(cls.mean3, (lv105_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv307 = R.call_tir(cls.expand_dims3, (lv306,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv308 = R.call_tir(cls.subtract3, (lv105_1, lv307), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv309 = R.call_tir(cls.variance3, (lv105_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv310 = R.call_tir(cls.expand_dims3, (lv309,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv311 = R.call_tir(cls.add9, (lv310,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv312 = R.call_tir(cls.tir_sqrt3, (lv311,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv313 = R.call_tir(cls.divide3, (lv308, lv312), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv314 = R.call_tir(cls.expand_dims3, (bn_gamma_17,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv315 = R.call_tir(cls.multiply3, (lv313, lv314), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv316 = R.call_tir(cls.expand_dims3, (bn_beta_17,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv317 = R.call_tir(cls.add10, (lv315, lv316), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv318 = R.call_tir(cls.multiply10, (bn_mm_17,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv319 = R.call_tir(cls.multiply11, (lv306,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv320 = R.call_tir(cls.add16, (lv318, lv319), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv321 = R.call_tir(cls.multiply10, (bn_mv_17,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv322 = R.call_tir(cls.multiply11, (lv309,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv323 = R.call_tir(cls.add16, (lv321, lv322), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv106_1: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv317, lv320, lv323
            lv107_1: R.Tensor((32, 512, 4, 4), dtype="float32") = lv106_1[0]
            lv108_1: R.Tensor((512,), dtype="float32") = lv106_1[1]
            lv109_1: R.Tensor((512,), dtype="float32") = lv106_1[2]
            lv110_1 = R.call_tir(cls.add11, (lv102_1, lv107_1), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv111_1 = R.call_tir(cls.relu3, (lv110_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv112_1 = R.call_tir(cls.conv2d9, (lv111_1, conv2d_weight_18), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv324 = R.call_tir(cls.mean3, (lv112_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv325 = R.call_tir(cls.expand_dims3, (lv324,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv326 = R.call_tir(cls.subtract3, (lv112_1, lv325), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv327 = R.call_tir(cls.variance3, (lv112_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv328 = R.call_tir(cls.expand_dims3, (lv327,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv329 = R.call_tir(cls.add9, (lv328,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv330 = R.call_tir(cls.tir_sqrt3, (lv329,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv331 = R.call_tir(cls.divide3, (lv326, lv330), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv332 = R.call_tir(cls.expand_dims3, (bn_gamma_18,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv333 = R.call_tir(cls.multiply3, (lv331, lv332), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv334 = R.call_tir(cls.expand_dims3, (bn_beta_18,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv335 = R.call_tir(cls.add10, (lv333, lv334), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv336 = R.call_tir(cls.multiply10, (bn_mm_18,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv337 = R.call_tir(cls.multiply11, (lv324,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv338 = R.call_tir(cls.add16, (lv336, lv337), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv339 = R.call_tir(cls.multiply10, (bn_mv_18,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv340 = R.call_tir(cls.multiply11, (lv327,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv341 = R.call_tir(cls.add16, (lv339, lv340), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv113_1: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv335, lv338, lv341
            lv114_1: R.Tensor((32, 512, 4, 4), dtype="float32") = lv113_1[0]
            lv115_1: R.Tensor((512,), dtype="float32") = lv113_1[1]
            lv116_1: R.Tensor((512,), dtype="float32") = lv113_1[2]
            lv117_1 = R.call_tir(cls.relu3, (lv114_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv118_1 = R.call_tir(cls.conv2d9, (lv117_1, conv2d_weight_19), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv342 = R.call_tir(cls.mean3, (lv118_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv343 = R.call_tir(cls.expand_dims3, (lv342,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv344 = R.call_tir(cls.subtract3, (lv118_1, lv343), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv345 = R.call_tir(cls.variance3, (lv118_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv346 = R.call_tir(cls.expand_dims3, (lv345,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv347 = R.call_tir(cls.add9, (lv346,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv348 = R.call_tir(cls.tir_sqrt3, (lv347,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv349 = R.call_tir(cls.divide3, (lv344, lv348), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv350 = R.call_tir(cls.expand_dims3, (bn_gamma_19,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv351 = R.call_tir(cls.multiply3, (lv349, lv350), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv352 = R.call_tir(cls.expand_dims3, (bn_beta_19,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv353 = R.call_tir(cls.add10, (lv351, lv352), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv354 = R.call_tir(cls.multiply10, (bn_mm_19,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv355 = R.call_tir(cls.multiply11, (lv342,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv356 = R.call_tir(cls.add16, (lv354, lv355), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv357 = R.call_tir(cls.multiply10, (bn_mv_19,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv358 = R.call_tir(cls.multiply11, (lv345,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv359 = R.call_tir(cls.add16, (lv357, lv358), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv119_1: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv353, lv356, lv359
            lv120_1: R.Tensor((32, 512, 4, 4), dtype="float32") = lv119_1[0]
            lv121_1: R.Tensor((512,), dtype="float32") = lv119_1[1]
            lv122_1: R.Tensor((512,), dtype="float32") = lv119_1[2]
            lv123_1 = R.call_tir(cls.add11, (lv120_1, lv111_1), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv124_1 = R.call_tir(cls.relu3, (lv123_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv125_1 = R.call_tir(cls.avg_pool2d, (lv124_1,), out_sinfo=R.Tensor((32, 512, 1, 1), dtype="float32"))
            lv126_1 = R.call_tir(cls.reshape, (lv125_1,), out_sinfo=R.Tensor((32, 512), dtype="float32"))
            lv127_1 = R.call_tir(cls.matmul, (lv126_1, ln_weight), out_sinfo=R.Tensor((32, 10), dtype="float32"))
            lv128_1 = R.call_tir(cls.add12, (lv127_1, ln_bias), out_sinfo=R.Tensor((32, 10), dtype="float32"))
            gv: R.Tensor((32, 10), dtype="float32") = lv128_1
            gv1: R.Tensor((64,), dtype="float32") = lv3_1
            gv2: R.Tensor((64,), dtype="float32") = lv4_1
            gv3: R.Tensor((64,), dtype="float32") = lv9_1
            gv4: R.Tensor((64,), dtype="float32") = lv10_1
            gv5: R.Tensor((64,), dtype="float32") = lv15_1
            gv6: R.Tensor((64,), dtype="float32") = lv16_1
            gv7: R.Tensor((64,), dtype="float32") = lv22_1
            gv8: R.Tensor((64,), dtype="float32") = lv23_1
            gv9: R.Tensor((64,), dtype="float32") = lv28_1
            gv10: R.Tensor((64,), dtype="float32") = lv29_1
            gv11: R.Tensor((128,), dtype="float32") = lv35_1
            gv12: R.Tensor((128,), dtype="float32") = lv36_1
            gv13: R.Tensor((128,), dtype="float32") = lv41_1
            gv14: R.Tensor((128,), dtype="float32") = lv42_1
            gv15: R.Tensor((128,), dtype="float32") = lv46_1
            gv16: R.Tensor((128,), dtype="float32") = lv47_1
            gv17: R.Tensor((128,), dtype="float32") = lv53_1
            gv18: R.Tensor((128,), dtype="float32") = lv54_1
            gv19: R.Tensor((128,), dtype="float32") = lv59_1
            gv20: R.Tensor((128,), dtype="float32") = lv60_1
            gv21: R.Tensor((256,), dtype="float32") = lv66_1
            gv22: R.Tensor((256,), dtype="float32") = lv67_1
            gv23: R.Tensor((256,), dtype="float32") = lv72_1
            gv24: R.Tensor((256,), dtype="float32") = lv73_1
            gv25: R.Tensor((256,), dtype="float32") = lv77_1
            gv26: R.Tensor((256,), dtype="float32") = lv78_1
            gv27: R.Tensor((256,), dtype="float32") = lv84_1
            gv28: R.Tensor((256,), dtype="float32") = lv85_1
            gv29: R.Tensor((256,), dtype="float32") = lv90_1
            gv30: R.Tensor((256,), dtype="float32") = lv91_1
            gv31: R.Tensor((512,), dtype="float32") = lv97_1
            gv32: R.Tensor((512,), dtype="float32") = lv98_1
            gv33: R.Tensor((512,), dtype="float32") = lv103_1
            gv34: R.Tensor((512,), dtype="float32") = lv104_1
            gv35: R.Tensor((512,), dtype="float32") = lv108_1
            gv36: R.Tensor((512,), dtype="float32") = lv109_1
            gv37: R.Tensor((512,), dtype="float32") = lv115_1
            gv38: R.Tensor((512,), dtype="float32") = lv116_1
            gv39: R.Tensor((512,), dtype="float32") = lv121_1
            gv40: R.Tensor((512,), dtype="float32") = lv122_1
            lv_2 = R.call_tir(cls.log_softmax, (gv,), out_sinfo=R.Tensor((32, 10), dtype="float32"))
            gv_1 = R.call_tir(cls.nll_loss_without_weight, (lv_2, targets), out_sinfo=R.Tensor((), dtype="float32"))
            R.output(gv1, gv2, gv3, gv4, gv5, gv6, gv7, gv8, gv9, gv10, gv11, gv12, gv13, gv14, gv15, gv16, gv17, gv18, gv19, gv20, gv21, gv22, gv23, gv24, gv25, gv26, gv27, gv28, gv29, gv30, gv31, gv32, gv33, gv34, gv35, gv36, gv37, gv38, gv39, gv40, gv_1)
        return (gv_1, gv1, gv2, gv3, gv4, gv5, gv6, gv7, gv8, gv9, gv10, gv11, gv12, gv13, gv14, gv15, gv16, gv17, gv18, gv19, gv20, gv21, gv22, gv23, gv24, gv25, gv26, gv27, gv28, gv29, gv30, gv31, gv32, gv33, gv34, gv35, gv36, gv37, gv38, gv39, gv40)

    @R.function
    def backbone_loss_adjoint(input: R.Tensor((32, 3, 32, 32), dtype="float32"), conv2d_weight: R.Tensor((64, 3, 3, 3), dtype="float32"), bn_gamma: R.Tensor((64,), dtype="float32"), bn_beta: R.Tensor((64,), dtype="float32"), conv2d_weight_1: R.Tensor((64, 64, 3, 3), dtype="float32"), bn_gamma_1: R.Tensor((64,), dtype="float32"), bn_beta_1: R.Tensor((64,), dtype="float32"), conv2d_weight_2: R.Tensor((64, 64, 3, 3), dtype="float32"), bn_gamma_2: R.Tensor((64,), dtype="float32"), bn_beta_2: R.Tensor((64,), dtype="float32"), conv2d_weight_3: R.Tensor((64, 64, 3, 3), dtype="float32"), bn_gamma_3: R.Tensor((64,), dtype="float32"), bn_beta_3: R.Tensor((64,), dtype="float32"), conv2d_weight_4: R.Tensor((64, 64, 3, 3), dtype="float32"), bn_gamma_4: R.Tensor((64,), dtype="float32"), bn_beta_4: R.Tensor((64,), dtype="float32"), conv2d_weight_5: R.Tensor((128, 64, 3, 3), dtype="float32"), bn_gamma_5: R.Tensor((128,), dtype="float32"), bn_beta_5: R.Tensor((128,), dtype="float32"), conv2d_weight_6: R.Tensor((128, 128, 3, 3), dtype="float32"), bn_gamma_6: R.Tensor((128,), dtype="float32"), bn_beta_6: R.Tensor((128,), dtype="float32"), conv2d_weight_7: R.Tensor((128, 64, 1, 1), dtype="float32"), bn_gamma_7: R.Tensor((128,), dtype="float32"), bn_beta_7: R.Tensor((128,), dtype="float32"), conv2d_weight_8: R.Tensor((128, 128, 3, 3), dtype="float32"), bn_gamma_8: R.Tensor((128,), dtype="float32"), bn_beta_8: R.Tensor((128,), dtype="float32"), conv2d_weight_9: R.Tensor((128, 128, 3, 3), dtype="float32"), bn_gamma_9: R.Tensor((128,), dtype="float32"), bn_beta_9: R.Tensor((128,), dtype="float32"), conv2d_weight_10: R.Tensor((256, 128, 3, 3), dtype="float32"), bn_gamma_10: R.Tensor((256,), dtype="float32"), bn_beta_10: R.Tensor((256,), dtype="float32"), conv2d_weight_11: R.Tensor((256, 256, 3, 3), dtype="float32"), bn_gamma_11: R.Tensor((256,), dtype="float32"), bn_beta_11: R.Tensor((256,), dtype="float32"), conv2d_weight_12: R.Tensor((256, 128, 1, 1), dtype="float32"), bn_gamma_12: R.Tensor((256,), dtype="float32"), bn_beta_12: R.Tensor((256,), dtype="float32"), conv2d_weight_13: R.Tensor((256, 256, 3, 3), dtype="float32"), bn_gamma_13: R.Tensor((256,), dtype="float32"), bn_beta_13: R.Tensor((256,), dtype="float32"), conv2d_weight_14: R.Tensor((256, 256, 3, 3), dtype="float32"), bn_gamma_14: R.Tensor((256,), dtype="float32"), bn_beta_14: R.Tensor((256,), dtype="float32"), conv2d_weight_15: R.Tensor((512, 256, 3, 3), dtype="float32"), bn_gamma_15: R.Tensor((512,), dtype="float32"), bn_beta_15: R.Tensor((512,), dtype="float32"), conv2d_weight_16: R.Tensor((512, 512, 3, 3), dtype="float32"), bn_gamma_16: R.Tensor((512,), dtype="float32"), bn_beta_16: R.Tensor((512,), dtype="float32"), conv2d_weight_17: R.Tensor((512, 256, 1, 1), dtype="float32"), bn_gamma_17: R.Tensor((512,), dtype="float32"), bn_beta_17: R.Tensor((512,), dtype="float32"), conv2d_weight_18: R.Tensor((512, 512, 3, 3), dtype="float32"), bn_gamma_18: R.Tensor((512,), dtype="float32"), bn_beta_18: R.Tensor((512,), dtype="float32"), conv2d_weight_19: R.Tensor((512, 512, 3, 3), dtype="float32"), bn_gamma_19: R.Tensor((512,), dtype="float32"), bn_beta_19: R.Tensor((512,), dtype="float32"), ln_weight: R.Tensor((512, 10), dtype="float32"), ln_bias: R.Tensor((10,), dtype="float32"), bn_mm: R.Tensor((64,), dtype="float32"), bn_mv: R.Tensor((64,), dtype="float32"), bn_mm_1: R.Tensor((64,), dtype="float32"), bn_mv_1: R.Tensor((64,), dtype="float32"), bn_mm_2: R.Tensor((64,), dtype="float32"), bn_mv_2: R.Tensor((64,), dtype="float32"), bn_mm_3: R.Tensor((64,), dtype="float32"), bn_mv_3: R.Tensor((64,), dtype="float32"), bn_mm_4: R.Tensor((64,), dtype="float32"), bn_mv_4: R.Tensor((64,), dtype="float32"), bn_mm_5: R.Tensor((128,), dtype="float32"), bn_mv_5: R.Tensor((128,), dtype="float32"), bn_mm_6: R.Tensor((128,), dtype="float32"), bn_mv_6: R.Tensor((128,), dtype="float32"), bn_mm_7: R.Tensor((128,), dtype="float32"), bn_mv_7: R.Tensor((128,), dtype="float32"), bn_mm_8: R.Tensor((128,), dtype="float32"), bn_mv_8: R.Tensor((128,), dtype="float32"), bn_mm_9: R.Tensor((128,), dtype="float32"), bn_mv_9: R.Tensor((128,), dtype="float32"), bn_mm_10: R.Tensor((256,), dtype="float32"), bn_mv_10: R.Tensor((256,), dtype="float32"), bn_mm_11: R.Tensor((256,), dtype="float32"), bn_mv_11: R.Tensor((256,), dtype="float32"), bn_mm_12: R.Tensor((256,), dtype="float32"), bn_mv_12: R.Tensor((256,), dtype="float32"), bn_mm_13: R.Tensor((256,), dtype="float32"), bn_mv_13: R.Tensor((256,), dtype="float32"), bn_mm_14: R.Tensor((256,), dtype="float32"), bn_mv_14: R.Tensor((256,), dtype="float32"), bn_mm_15: R.Tensor((512,), dtype="float32"), bn_mv_15: R.Tensor((512,), dtype="float32"), bn_mm_16: R.Tensor((512,), dtype="float32"), bn_mv_16: R.Tensor((512,), dtype="float32"), bn_mm_17: R.Tensor((512,), dtype="float32"), bn_mv_17: R.Tensor((512,), dtype="float32"), bn_mm_18: R.Tensor((512,), dtype="float32"), bn_mv_18: R.Tensor((512,), dtype="float32"), bn_mm_19: R.Tensor((512,), dtype="float32"), bn_mv_19: R.Tensor((512,), dtype="float32"), targets: R.Tensor((32,), dtype="int64")) -> R.Tuple(R.Tuple(R.Tensor((), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")), R.Tuple(R.Tensor((64, 3, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((128, 64, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 64, 1, 1), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((256, 128, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 128, 1, 1), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((512, 256, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 256, 1, 1), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 10), dtype="float32"), R.Tensor((10,), dtype="float32"))):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.conv2d, (input, conv2d_weight), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv_1 = R.call_tir(cls.mean, (lv,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv1 = R.call_tir(cls.expand_dims, (lv_1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv2 = R.call_tir(cls.subtract, (lv, lv1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv3 = R.call_tir(cls.variance, (lv,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv4 = R.call_tir(cls.expand_dims, (lv3,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv5 = R.call_tir(cls.add, (lv4,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv6 = R.call_tir(cls.tir_sqrt, (lv5,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv7 = R.call_tir(cls.divide, (lv2, lv6), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv8 = R.call_tir(cls.expand_dims, (bn_gamma,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv9 = R.call_tir(cls.multiply, (lv7, lv8), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv10 = R.call_tir(cls.expand_dims, (bn_beta,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv11 = R.call_tir(cls.add1, (lv9, lv10), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv12 = R.call_tir(cls.multiply4, (bn_mm,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv13 = R.call_tir(cls.multiply5, (lv_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv14 = R.call_tir(cls.add13, (lv12, lv13), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv15 = R.call_tir(cls.multiply4, (bn_mv,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv16 = R.call_tir(cls.multiply5, (lv3,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv17 = R.call_tir(cls.add13, (lv15, lv16), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv1_1: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv11, lv14, lv17
            lv2_1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv1_1[0]
            lv3_1: R.Tensor((64,), dtype="float32") = lv1_1[1]
            lv4_1: R.Tensor((64,), dtype="float32") = lv1_1[2]
            lv5_1 = R.call_tir(cls.relu, (lv2_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv6_1 = R.call_tir(cls.conv2d1, (lv5_1, conv2d_weight_1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv18 = R.call_tir(cls.mean, (lv6_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv19 = R.call_tir(cls.expand_dims, (lv18,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv20 = R.call_tir(cls.subtract, (lv6_1, lv19), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv21 = R.call_tir(cls.variance, (lv6_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv22 = R.call_tir(cls.expand_dims, (lv21,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv23 = R.call_tir(cls.add, (lv22,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv24 = R.call_tir(cls.tir_sqrt, (lv23,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv25 = R.call_tir(cls.divide, (lv20, lv24), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv26 = R.call_tir(cls.expand_dims, (bn_gamma_1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv27 = R.call_tir(cls.multiply, (lv25, lv26), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv28 = R.call_tir(cls.expand_dims, (bn_beta_1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv29 = R.call_tir(cls.add1, (lv27, lv28), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv30 = R.call_tir(cls.multiply4, (bn_mm_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv31 = R.call_tir(cls.multiply5, (lv18,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv32 = R.call_tir(cls.add13, (lv30, lv31), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv33 = R.call_tir(cls.multiply4, (bn_mv_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv34 = R.call_tir(cls.multiply5, (lv21,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv35 = R.call_tir(cls.add13, (lv33, lv34), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv7_1: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv29, lv32, lv35
            lv8_1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv7_1[0]
            lv9_1: R.Tensor((64,), dtype="float32") = lv7_1[1]
            lv10_1: R.Tensor((64,), dtype="float32") = lv7_1[2]
            lv11_1 = R.call_tir(cls.relu, (lv8_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv12_1 = R.call_tir(cls.conv2d1, (lv11_1, conv2d_weight_2), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv36 = R.call_tir(cls.mean, (lv12_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv37 = R.call_tir(cls.expand_dims, (lv36,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv38 = R.call_tir(cls.subtract, (lv12_1, lv37), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv39 = R.call_tir(cls.variance, (lv12_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv40 = R.call_tir(cls.expand_dims, (lv39,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv41 = R.call_tir(cls.add, (lv40,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv42 = R.call_tir(cls.tir_sqrt, (lv41,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv43 = R.call_tir(cls.divide, (lv38, lv42), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv44 = R.call_tir(cls.expand_dims, (bn_gamma_2,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv45 = R.call_tir(cls.multiply, (lv43, lv44), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv46 = R.call_tir(cls.expand_dims, (bn_beta_2,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv47 = R.call_tir(cls.add1, (lv45, lv46), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv48 = R.call_tir(cls.multiply4, (bn_mm_2,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv49 = R.call_tir(cls.multiply5, (lv36,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv50 = R.call_tir(cls.add13, (lv48, lv49), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv51 = R.call_tir(cls.multiply4, (bn_mv_2,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv52 = R.call_tir(cls.multiply5, (lv39,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv53 = R.call_tir(cls.add13, (lv51, lv52), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv13_1: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv47, lv50, lv53
            lv14_1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv13_1[0]
            lv15_1: R.Tensor((64,), dtype="float32") = lv13_1[1]
            lv16_1: R.Tensor((64,), dtype="float32") = lv13_1[2]
            lv17_1 = R.call_tir(cls.add2, (lv14_1, lv5_1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv18_1 = R.call_tir(cls.relu, (lv17_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv19_1 = R.call_tir(cls.conv2d1, (lv18_1, conv2d_weight_3), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv54 = R.call_tir(cls.mean, (lv19_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv55 = R.call_tir(cls.expand_dims, (lv54,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv56 = R.call_tir(cls.subtract, (lv19_1, lv55), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv57 = R.call_tir(cls.variance, (lv19_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv58 = R.call_tir(cls.expand_dims, (lv57,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv59 = R.call_tir(cls.add, (lv58,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv60 = R.call_tir(cls.tir_sqrt, (lv59,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv61 = R.call_tir(cls.divide, (lv56, lv60), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv62 = R.call_tir(cls.expand_dims, (bn_gamma_3,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv63 = R.call_tir(cls.multiply, (lv61, lv62), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv64 = R.call_tir(cls.expand_dims, (bn_beta_3,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv65 = R.call_tir(cls.add1, (lv63, lv64), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv66 = R.call_tir(cls.multiply4, (bn_mm_3,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv67 = R.call_tir(cls.multiply5, (lv54,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv68 = R.call_tir(cls.add13, (lv66, lv67), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv69 = R.call_tir(cls.multiply4, (bn_mv_3,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv70 = R.call_tir(cls.multiply5, (lv57,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv71 = R.call_tir(cls.add13, (lv69, lv70), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv20_1: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv65, lv68, lv71
            lv21_1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv20_1[0]
            lv22_1: R.Tensor((64,), dtype="float32") = lv20_1[1]
            lv23_1: R.Tensor((64,), dtype="float32") = lv20_1[2]
            lv24_1 = R.call_tir(cls.relu, (lv21_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv25_1 = R.call_tir(cls.conv2d1, (lv24_1, conv2d_weight_4), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv72 = R.call_tir(cls.mean, (lv25_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv73 = R.call_tir(cls.expand_dims, (lv72,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv74 = R.call_tir(cls.subtract, (lv25_1, lv73), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv75 = R.call_tir(cls.variance, (lv25_1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv76 = R.call_tir(cls.expand_dims, (lv75,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv77 = R.call_tir(cls.add, (lv76,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv78 = R.call_tir(cls.tir_sqrt, (lv77,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv79 = R.call_tir(cls.divide, (lv74, lv78), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv80 = R.call_tir(cls.expand_dims, (bn_gamma_4,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv81 = R.call_tir(cls.multiply, (lv79, lv80), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv82 = R.call_tir(cls.expand_dims, (bn_beta_4,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv83 = R.call_tir(cls.add1, (lv81, lv82), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv84 = R.call_tir(cls.multiply4, (bn_mm_4,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv85 = R.call_tir(cls.multiply5, (lv72,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv86 = R.call_tir(cls.add13, (lv84, lv85), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv87 = R.call_tir(cls.multiply4, (bn_mv_4,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv88 = R.call_tir(cls.multiply5, (lv75,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv89 = R.call_tir(cls.add13, (lv87, lv88), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv26_1: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv83, lv86, lv89
            lv27_1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv26_1[0]
            lv28_1: R.Tensor((64,), dtype="float32") = lv26_1[1]
            lv29_1: R.Tensor((64,), dtype="float32") = lv26_1[2]
            lv30_1 = R.call_tir(cls.add2, (lv27_1, lv18_1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv31_1 = R.call_tir(cls.relu, (lv30_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv32_1 = R.call_tir(cls.conv2d2, (lv31_1, conv2d_weight_5), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv90 = R.call_tir(cls.mean1, (lv32_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv91 = R.call_tir(cls.expand_dims1, (lv90,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv92 = R.call_tir(cls.subtract1, (lv32_1, lv91), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv93 = R.call_tir(cls.variance1, (lv32_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv94 = R.call_tir(cls.expand_dims1, (lv93,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv95 = R.call_tir(cls.add3, (lv94,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv96 = R.call_tir(cls.tir_sqrt1, (lv95,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv97 = R.call_tir(cls.divide1, (lv92, lv96), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv98 = R.call_tir(cls.expand_dims1, (bn_gamma_5,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv99 = R.call_tir(cls.multiply1, (lv97, lv98), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv100 = R.call_tir(cls.expand_dims1, (bn_beta_5,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv101 = R.call_tir(cls.add4, (lv99, lv100), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv102 = R.call_tir(cls.multiply6, (bn_mm_5,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv103 = R.call_tir(cls.multiply7, (lv90,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv104 = R.call_tir(cls.add14, (lv102, lv103), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv105 = R.call_tir(cls.multiply6, (bn_mv_5,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv106 = R.call_tir(cls.multiply7, (lv93,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv107 = R.call_tir(cls.add14, (lv105, lv106), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv33_1: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv101, lv104, lv107
            lv34_1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv33_1[0]
            lv35_1: R.Tensor((128,), dtype="float32") = lv33_1[1]
            lv36_1: R.Tensor((128,), dtype="float32") = lv33_1[2]
            lv37_1 = R.call_tir(cls.relu1, (lv34_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv38_1 = R.call_tir(cls.conv2d3, (lv37_1, conv2d_weight_6), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv108 = R.call_tir(cls.mean1, (lv38_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv109 = R.call_tir(cls.expand_dims1, (lv108,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv110 = R.call_tir(cls.subtract1, (lv38_1, lv109), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv111 = R.call_tir(cls.variance1, (lv38_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv112 = R.call_tir(cls.expand_dims1, (lv111,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv113 = R.call_tir(cls.add3, (lv112,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv114 = R.call_tir(cls.tir_sqrt1, (lv113,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv115 = R.call_tir(cls.divide1, (lv110, lv114), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv116 = R.call_tir(cls.expand_dims1, (bn_gamma_6,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv117 = R.call_tir(cls.multiply1, (lv115, lv116), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv118 = R.call_tir(cls.expand_dims1, (bn_beta_6,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv119 = R.call_tir(cls.add4, (lv117, lv118), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv120 = R.call_tir(cls.multiply6, (bn_mm_6,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv121 = R.call_tir(cls.multiply7, (lv108,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv122 = R.call_tir(cls.add14, (lv120, lv121), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv123 = R.call_tir(cls.multiply6, (bn_mv_6,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv124 = R.call_tir(cls.multiply7, (lv111,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv125 = R.call_tir(cls.add14, (lv123, lv124), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv39_1: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv119, lv122, lv125
            lv40_1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv39_1[0]
            lv41_1: R.Tensor((128,), dtype="float32") = lv39_1[1]
            lv42_1: R.Tensor((128,), dtype="float32") = lv39_1[2]
            lv43_1 = R.call_tir(cls.conv2d4, (lv31_1, conv2d_weight_7), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv126 = R.call_tir(cls.mean1, (lv43_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv127 = R.call_tir(cls.expand_dims1, (lv126,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv128 = R.call_tir(cls.subtract1, (lv43_1, lv127), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv129 = R.call_tir(cls.variance1, (lv43_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv130 = R.call_tir(cls.expand_dims1, (lv129,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv131 = R.call_tir(cls.add3, (lv130,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv132 = R.call_tir(cls.tir_sqrt1, (lv131,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv133 = R.call_tir(cls.divide1, (lv128, lv132), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv134 = R.call_tir(cls.expand_dims1, (bn_gamma_7,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv135 = R.call_tir(cls.multiply1, (lv133, lv134), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv136 = R.call_tir(cls.expand_dims1, (bn_beta_7,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv137 = R.call_tir(cls.add4, (lv135, lv136), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv138 = R.call_tir(cls.multiply6, (bn_mm_7,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv139 = R.call_tir(cls.multiply7, (lv126,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv140 = R.call_tir(cls.add14, (lv138, lv139), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv141 = R.call_tir(cls.multiply6, (bn_mv_7,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv142 = R.call_tir(cls.multiply7, (lv129,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv143 = R.call_tir(cls.add14, (lv141, lv142), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv44_1: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv137, lv140, lv143
            lv45_1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv44_1[0]
            lv46_1: R.Tensor((128,), dtype="float32") = lv44_1[1]
            lv47_1: R.Tensor((128,), dtype="float32") = lv44_1[2]
            lv48_1 = R.call_tir(cls.add5, (lv40_1, lv45_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv49_1 = R.call_tir(cls.relu1, (lv48_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv50_1 = R.call_tir(cls.conv2d3, (lv49_1, conv2d_weight_8), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv144 = R.call_tir(cls.mean1, (lv50_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv145 = R.call_tir(cls.expand_dims1, (lv144,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv146 = R.call_tir(cls.subtract1, (lv50_1, lv145), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv147 = R.call_tir(cls.variance1, (lv50_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv148 = R.call_tir(cls.expand_dims1, (lv147,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv149 = R.call_tir(cls.add3, (lv148,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv150 = R.call_tir(cls.tir_sqrt1, (lv149,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv151 = R.call_tir(cls.divide1, (lv146, lv150), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv152 = R.call_tir(cls.expand_dims1, (bn_gamma_8,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv153 = R.call_tir(cls.multiply1, (lv151, lv152), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv154 = R.call_tir(cls.expand_dims1, (bn_beta_8,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv155 = R.call_tir(cls.add4, (lv153, lv154), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv156 = R.call_tir(cls.multiply6, (bn_mm_8,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv157 = R.call_tir(cls.multiply7, (lv144,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv158 = R.call_tir(cls.add14, (lv156, lv157), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv159 = R.call_tir(cls.multiply6, (bn_mv_8,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv160 = R.call_tir(cls.multiply7, (lv147,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv161 = R.call_tir(cls.add14, (lv159, lv160), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv51_1: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv155, lv158, lv161
            lv52_1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv51_1[0]
            lv53_1: R.Tensor((128,), dtype="float32") = lv51_1[1]
            lv54_1: R.Tensor((128,), dtype="float32") = lv51_1[2]
            lv55_1 = R.call_tir(cls.relu1, (lv52_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv56_1 = R.call_tir(cls.conv2d3, (lv55_1, conv2d_weight_9), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv162 = R.call_tir(cls.mean1, (lv56_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv163 = R.call_tir(cls.expand_dims1, (lv162,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv164 = R.call_tir(cls.subtract1, (lv56_1, lv163), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv165 = R.call_tir(cls.variance1, (lv56_1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv166 = R.call_tir(cls.expand_dims1, (lv165,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv167 = R.call_tir(cls.add3, (lv166,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv168 = R.call_tir(cls.tir_sqrt1, (lv167,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv169 = R.call_tir(cls.divide1, (lv164, lv168), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv170 = R.call_tir(cls.expand_dims1, (bn_gamma_9,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv171 = R.call_tir(cls.multiply1, (lv169, lv170), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv172 = R.call_tir(cls.expand_dims1, (bn_beta_9,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv173 = R.call_tir(cls.add4, (lv171, lv172), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv174 = R.call_tir(cls.multiply6, (bn_mm_9,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv175 = R.call_tir(cls.multiply7, (lv162,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv176 = R.call_tir(cls.add14, (lv174, lv175), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv177 = R.call_tir(cls.multiply6, (bn_mv_9,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv178 = R.call_tir(cls.multiply7, (lv165,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv179 = R.call_tir(cls.add14, (lv177, lv178), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv57_1: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv173, lv176, lv179
            lv58_1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv57_1[0]
            lv59_1: R.Tensor((128,), dtype="float32") = lv57_1[1]
            lv60_1: R.Tensor((128,), dtype="float32") = lv57_1[2]
            lv61_1 = R.call_tir(cls.add5, (lv58_1, lv49_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv62_1 = R.call_tir(cls.relu1, (lv61_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv63_1 = R.call_tir(cls.conv2d5, (lv62_1, conv2d_weight_10), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv180 = R.call_tir(cls.mean2, (lv63_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv181 = R.call_tir(cls.expand_dims2, (lv180,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv182 = R.call_tir(cls.subtract2, (lv63_1, lv181), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv183 = R.call_tir(cls.variance2, (lv63_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv184 = R.call_tir(cls.expand_dims2, (lv183,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv185 = R.call_tir(cls.add6, (lv184,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv186 = R.call_tir(cls.tir_sqrt2, (lv185,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv187 = R.call_tir(cls.divide2, (lv182, lv186), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv188 = R.call_tir(cls.expand_dims2, (bn_gamma_10,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv189 = R.call_tir(cls.multiply2, (lv187, lv188), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv190 = R.call_tir(cls.expand_dims2, (bn_beta_10,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv191 = R.call_tir(cls.add7, (lv189, lv190), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv192 = R.call_tir(cls.multiply8, (bn_mm_10,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv193 = R.call_tir(cls.multiply9, (lv180,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv194 = R.call_tir(cls.add15, (lv192, lv193), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv195 = R.call_tir(cls.multiply8, (bn_mv_10,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv196 = R.call_tir(cls.multiply9, (lv183,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv197 = R.call_tir(cls.add15, (lv195, lv196), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv64_1: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv191, lv194, lv197
            lv65_1: R.Tensor((32, 256, 8, 8), dtype="float32") = lv64_1[0]
            lv66_1: R.Tensor((256,), dtype="float32") = lv64_1[1]
            lv67_1: R.Tensor((256,), dtype="float32") = lv64_1[2]
            lv68_1 = R.call_tir(cls.relu2, (lv65_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv69_1 = R.call_tir(cls.conv2d6, (lv68_1, conv2d_weight_11), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv198 = R.call_tir(cls.mean2, (lv69_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv199 = R.call_tir(cls.expand_dims2, (lv198,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv200 = R.call_tir(cls.subtract2, (lv69_1, lv199), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv201 = R.call_tir(cls.variance2, (lv69_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv202 = R.call_tir(cls.expand_dims2, (lv201,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv203 = R.call_tir(cls.add6, (lv202,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv204 = R.call_tir(cls.tir_sqrt2, (lv203,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv205 = R.call_tir(cls.divide2, (lv200, lv204), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv206 = R.call_tir(cls.expand_dims2, (bn_gamma_11,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv207 = R.call_tir(cls.multiply2, (lv205, lv206), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv208 = R.call_tir(cls.expand_dims2, (bn_beta_11,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv209 = R.call_tir(cls.add7, (lv207, lv208), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv210 = R.call_tir(cls.multiply8, (bn_mm_11,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv211 = R.call_tir(cls.multiply9, (lv198,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv212 = R.call_tir(cls.add15, (lv210, lv211), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv213 = R.call_tir(cls.multiply8, (bn_mv_11,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv214 = R.call_tir(cls.multiply9, (lv201,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv215 = R.call_tir(cls.add15, (lv213, lv214), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv70_1: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv209, lv212, lv215
            lv71_1: R.Tensor((32, 256, 8, 8), dtype="float32") = lv70_1[0]
            lv72_1: R.Tensor((256,), dtype="float32") = lv70_1[1]
            lv73_1: R.Tensor((256,), dtype="float32") = lv70_1[2]
            lv74_1 = R.call_tir(cls.conv2d7, (lv62_1, conv2d_weight_12), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv216 = R.call_tir(cls.mean2, (lv74_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv217 = R.call_tir(cls.expand_dims2, (lv216,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv218 = R.call_tir(cls.subtract2, (lv74_1, lv217), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv219 = R.call_tir(cls.variance2, (lv74_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv220 = R.call_tir(cls.expand_dims2, (lv219,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv221 = R.call_tir(cls.add6, (lv220,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv222 = R.call_tir(cls.tir_sqrt2, (lv221,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv223 = R.call_tir(cls.divide2, (lv218, lv222), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv224 = R.call_tir(cls.expand_dims2, (bn_gamma_12,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv225 = R.call_tir(cls.multiply2, (lv223, lv224), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv226 = R.call_tir(cls.expand_dims2, (bn_beta_12,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv227 = R.call_tir(cls.add7, (lv225, lv226), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv228 = R.call_tir(cls.multiply8, (bn_mm_12,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv229 = R.call_tir(cls.multiply9, (lv216,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv230 = R.call_tir(cls.add15, (lv228, lv229), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv231 = R.call_tir(cls.multiply8, (bn_mv_12,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv232 = R.call_tir(cls.multiply9, (lv219,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv233 = R.call_tir(cls.add15, (lv231, lv232), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv75_1: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv227, lv230, lv233
            lv76_1: R.Tensor((32, 256, 8, 8), dtype="float32") = lv75_1[0]
            lv77_1: R.Tensor((256,), dtype="float32") = lv75_1[1]
            lv78_1: R.Tensor((256,), dtype="float32") = lv75_1[2]
            lv79_1 = R.call_tir(cls.add8, (lv71_1, lv76_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv80_1 = R.call_tir(cls.relu2, (lv79_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv81_1 = R.call_tir(cls.conv2d6, (lv80_1, conv2d_weight_13), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv234 = R.call_tir(cls.mean2, (lv81_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv235 = R.call_tir(cls.expand_dims2, (lv234,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv236 = R.call_tir(cls.subtract2, (lv81_1, lv235), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv237 = R.call_tir(cls.variance2, (lv81_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv238 = R.call_tir(cls.expand_dims2, (lv237,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv239 = R.call_tir(cls.add6, (lv238,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv240 = R.call_tir(cls.tir_sqrt2, (lv239,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv241 = R.call_tir(cls.divide2, (lv236, lv240), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv242 = R.call_tir(cls.expand_dims2, (bn_gamma_13,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv243 = R.call_tir(cls.multiply2, (lv241, lv242), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv244 = R.call_tir(cls.expand_dims2, (bn_beta_13,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv245 = R.call_tir(cls.add7, (lv243, lv244), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv246 = R.call_tir(cls.multiply8, (bn_mm_13,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv247 = R.call_tir(cls.multiply9, (lv234,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv248 = R.call_tir(cls.add15, (lv246, lv247), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv249 = R.call_tir(cls.multiply8, (bn_mv_13,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv250 = R.call_tir(cls.multiply9, (lv237,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv251 = R.call_tir(cls.add15, (lv249, lv250), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv82_1: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv245, lv248, lv251
            lv83_1: R.Tensor((32, 256, 8, 8), dtype="float32") = lv82_1[0]
            lv84_1: R.Tensor((256,), dtype="float32") = lv82_1[1]
            lv85_1: R.Tensor((256,), dtype="float32") = lv82_1[2]
            lv86_1 = R.call_tir(cls.relu2, (lv83_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv87_1 = R.call_tir(cls.conv2d6, (lv86_1, conv2d_weight_14), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv252 = R.call_tir(cls.mean2, (lv87_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv253 = R.call_tir(cls.expand_dims2, (lv252,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv254 = R.call_tir(cls.subtract2, (lv87_1, lv253), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv255 = R.call_tir(cls.variance2, (lv87_1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv256 = R.call_tir(cls.expand_dims2, (lv255,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv257 = R.call_tir(cls.add6, (lv256,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv258 = R.call_tir(cls.tir_sqrt2, (lv257,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv259 = R.call_tir(cls.divide2, (lv254, lv258), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv260 = R.call_tir(cls.expand_dims2, (bn_gamma_14,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv261 = R.call_tir(cls.multiply2, (lv259, lv260), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv262 = R.call_tir(cls.expand_dims2, (bn_beta_14,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv263 = R.call_tir(cls.add7, (lv261, lv262), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv264 = R.call_tir(cls.multiply8, (bn_mm_14,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv265 = R.call_tir(cls.multiply9, (lv252,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv266 = R.call_tir(cls.add15, (lv264, lv265), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv267 = R.call_tir(cls.multiply8, (bn_mv_14,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv268 = R.call_tir(cls.multiply9, (lv255,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv269 = R.call_tir(cls.add15, (lv267, lv268), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv88_1: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv263, lv266, lv269
            lv89_1: R.Tensor((32, 256, 8, 8), dtype="float32") = lv88_1[0]
            lv90_1: R.Tensor((256,), dtype="float32") = lv88_1[1]
            lv91_1: R.Tensor((256,), dtype="float32") = lv88_1[2]
            lv92_1 = R.call_tir(cls.add8, (lv89_1, lv80_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv93_1 = R.call_tir(cls.relu2, (lv92_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv94_1 = R.call_tir(cls.conv2d8, (lv93_1, conv2d_weight_15), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv270 = R.call_tir(cls.mean3, (lv94_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv271 = R.call_tir(cls.expand_dims3, (lv270,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv272 = R.call_tir(cls.subtract3, (lv94_1, lv271), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv273 = R.call_tir(cls.variance3, (lv94_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv274 = R.call_tir(cls.expand_dims3, (lv273,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv275 = R.call_tir(cls.add9, (lv274,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv276 = R.call_tir(cls.tir_sqrt3, (lv275,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv277 = R.call_tir(cls.divide3, (lv272, lv276), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv278 = R.call_tir(cls.expand_dims3, (bn_gamma_15,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv279 = R.call_tir(cls.multiply3, (lv277, lv278), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv280 = R.call_tir(cls.expand_dims3, (bn_beta_15,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv281 = R.call_tir(cls.add10, (lv279, lv280), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv282 = R.call_tir(cls.multiply10, (bn_mm_15,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv283 = R.call_tir(cls.multiply11, (lv270,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv284 = R.call_tir(cls.add16, (lv282, lv283), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv285 = R.call_tir(cls.multiply10, (bn_mv_15,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv286 = R.call_tir(cls.multiply11, (lv273,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv287 = R.call_tir(cls.add16, (lv285, lv286), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv95_1: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv281, lv284, lv287
            lv96_1: R.Tensor((32, 512, 4, 4), dtype="float32") = lv95_1[0]
            lv97_1: R.Tensor((512,), dtype="float32") = lv95_1[1]
            lv98_1: R.Tensor((512,), dtype="float32") = lv95_1[2]
            lv99_1 = R.call_tir(cls.relu3, (lv96_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv100_1 = R.call_tir(cls.conv2d9, (lv99_1, conv2d_weight_16), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv288 = R.call_tir(cls.mean3, (lv100_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv289 = R.call_tir(cls.expand_dims3, (lv288,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv290 = R.call_tir(cls.subtract3, (lv100_1, lv289), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv291 = R.call_tir(cls.variance3, (lv100_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv292 = R.call_tir(cls.expand_dims3, (lv291,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv293 = R.call_tir(cls.add9, (lv292,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv294 = R.call_tir(cls.tir_sqrt3, (lv293,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv295 = R.call_tir(cls.divide3, (lv290, lv294), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv296 = R.call_tir(cls.expand_dims3, (bn_gamma_16,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv297 = R.call_tir(cls.multiply3, (lv295, lv296), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv298 = R.call_tir(cls.expand_dims3, (bn_beta_16,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv299 = R.call_tir(cls.add10, (lv297, lv298), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv300 = R.call_tir(cls.multiply10, (bn_mm_16,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv301 = R.call_tir(cls.multiply11, (lv288,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv302 = R.call_tir(cls.add16, (lv300, lv301), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv303 = R.call_tir(cls.multiply10, (bn_mv_16,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv304 = R.call_tir(cls.multiply11, (lv291,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv305 = R.call_tir(cls.add16, (lv303, lv304), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv101_1: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv299, lv302, lv305
            lv102_1: R.Tensor((32, 512, 4, 4), dtype="float32") = lv101_1[0]
            lv103_1: R.Tensor((512,), dtype="float32") = lv101_1[1]
            lv104_1: R.Tensor((512,), dtype="float32") = lv101_1[2]
            lv105_1 = R.call_tir(cls.conv2d10, (lv93_1, conv2d_weight_17), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv306 = R.call_tir(cls.mean3, (lv105_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv307 = R.call_tir(cls.expand_dims3, (lv306,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv308 = R.call_tir(cls.subtract3, (lv105_1, lv307), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv309 = R.call_tir(cls.variance3, (lv105_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv310 = R.call_tir(cls.expand_dims3, (lv309,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv311 = R.call_tir(cls.add9, (lv310,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv312 = R.call_tir(cls.tir_sqrt3, (lv311,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv313 = R.call_tir(cls.divide3, (lv308, lv312), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv314 = R.call_tir(cls.expand_dims3, (bn_gamma_17,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv315 = R.call_tir(cls.multiply3, (lv313, lv314), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv316 = R.call_tir(cls.expand_dims3, (bn_beta_17,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv317 = R.call_tir(cls.add10, (lv315, lv316), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv318 = R.call_tir(cls.multiply10, (bn_mm_17,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv319 = R.call_tir(cls.multiply11, (lv306,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv320 = R.call_tir(cls.add16, (lv318, lv319), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv321 = R.call_tir(cls.multiply10, (bn_mv_17,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv322 = R.call_tir(cls.multiply11, (lv309,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv323 = R.call_tir(cls.add16, (lv321, lv322), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv106_1: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv317, lv320, lv323
            lv107_1: R.Tensor((32, 512, 4, 4), dtype="float32") = lv106_1[0]
            lv108_1: R.Tensor((512,), dtype="float32") = lv106_1[1]
            lv109_1: R.Tensor((512,), dtype="float32") = lv106_1[2]
            lv110_1 = R.call_tir(cls.add11, (lv102_1, lv107_1), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv111_1 = R.call_tir(cls.relu3, (lv110_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv112_1 = R.call_tir(cls.conv2d9, (lv111_1, conv2d_weight_18), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv324 = R.call_tir(cls.mean3, (lv112_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv325 = R.call_tir(cls.expand_dims3, (lv324,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv326 = R.call_tir(cls.subtract3, (lv112_1, lv325), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv327 = R.call_tir(cls.variance3, (lv112_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv328 = R.call_tir(cls.expand_dims3, (lv327,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv329 = R.call_tir(cls.add9, (lv328,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv330 = R.call_tir(cls.tir_sqrt3, (lv329,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv331 = R.call_tir(cls.divide3, (lv326, lv330), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv332 = R.call_tir(cls.expand_dims3, (bn_gamma_18,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv333 = R.call_tir(cls.multiply3, (lv331, lv332), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv334 = R.call_tir(cls.expand_dims3, (bn_beta_18,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv335 = R.call_tir(cls.add10, (lv333, lv334), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv336 = R.call_tir(cls.multiply10, (bn_mm_18,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv337 = R.call_tir(cls.multiply11, (lv324,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv338 = R.call_tir(cls.add16, (lv336, lv337), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv339 = R.call_tir(cls.multiply10, (bn_mv_18,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv340 = R.call_tir(cls.multiply11, (lv327,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv341 = R.call_tir(cls.add16, (lv339, lv340), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv113_1: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv335, lv338, lv341
            lv114_1: R.Tensor((32, 512, 4, 4), dtype="float32") = lv113_1[0]
            lv115_1: R.Tensor((512,), dtype="float32") = lv113_1[1]
            lv116_1: R.Tensor((512,), dtype="float32") = lv113_1[2]
            lv117_1 = R.call_tir(cls.relu3, (lv114_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv118_1 = R.call_tir(cls.conv2d9, (lv117_1, conv2d_weight_19), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv342 = R.call_tir(cls.mean3, (lv118_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv343 = R.call_tir(cls.expand_dims3, (lv342,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv344 = R.call_tir(cls.subtract3, (lv118_1, lv343), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv345 = R.call_tir(cls.variance3, (lv118_1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv346 = R.call_tir(cls.expand_dims3, (lv345,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv347 = R.call_tir(cls.add9, (lv346,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv348 = R.call_tir(cls.tir_sqrt3, (lv347,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv349 = R.call_tir(cls.divide3, (lv344, lv348), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv350 = R.call_tir(cls.expand_dims3, (bn_gamma_19,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv351 = R.call_tir(cls.multiply3, (lv349, lv350), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv352 = R.call_tir(cls.expand_dims3, (bn_beta_19,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv353 = R.call_tir(cls.add10, (lv351, lv352), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv354 = R.call_tir(cls.multiply10, (bn_mm_19,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv355 = R.call_tir(cls.multiply11, (lv342,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv356 = R.call_tir(cls.add16, (lv354, lv355), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv357 = R.call_tir(cls.multiply10, (bn_mv_19,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv358 = R.call_tir(cls.multiply11, (lv345,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv359 = R.call_tir(cls.add16, (lv357, lv358), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv119_1: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv353, lv356, lv359
            lv120_1: R.Tensor((32, 512, 4, 4), dtype="float32") = lv119_1[0]
            lv121_1: R.Tensor((512,), dtype="float32") = lv119_1[1]
            lv122_1: R.Tensor((512,), dtype="float32") = lv119_1[2]
            lv123_1 = R.call_tir(cls.add11, (lv120_1, lv111_1), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv124_1 = R.call_tir(cls.relu3, (lv123_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv125_1 = R.call_tir(cls.avg_pool2d, (lv124_1,), out_sinfo=R.Tensor((32, 512, 1, 1), dtype="float32"))
            lv126_1 = R.call_tir(cls.reshape, (lv125_1,), out_sinfo=R.Tensor((32, 512), dtype="float32"))
            lv127_1 = R.call_tir(cls.matmul, (lv126_1, ln_weight), out_sinfo=R.Tensor((32, 10), dtype="float32"))
            lv128_1 = R.call_tir(cls.add12, (lv127_1, ln_bias), out_sinfo=R.Tensor((32, 10), dtype="float32"))
            gv: R.Tensor((32, 10), dtype="float32") = lv128_1
            gv1: R.Tensor((64,), dtype="float32") = lv3_1
            gv2: R.Tensor((64,), dtype="float32") = lv4_1
            gv3: R.Tensor((64,), dtype="float32") = lv9_1
            gv4: R.Tensor((64,), dtype="float32") = lv10_1
            gv5: R.Tensor((64,), dtype="float32") = lv15_1
            gv6: R.Tensor((64,), dtype="float32") = lv16_1
            gv7: R.Tensor((64,), dtype="float32") = lv22_1
            gv8: R.Tensor((64,), dtype="float32") = lv23_1
            gv9: R.Tensor((64,), dtype="float32") = lv28_1
            gv10: R.Tensor((64,), dtype="float32") = lv29_1
            gv11: R.Tensor((128,), dtype="float32") = lv35_1
            gv12: R.Tensor((128,), dtype="float32") = lv36_1
            gv13: R.Tensor((128,), dtype="float32") = lv41_1
            gv14: R.Tensor((128,), dtype="float32") = lv42_1
            gv15: R.Tensor((128,), dtype="float32") = lv46_1
            gv16: R.Tensor((128,), dtype="float32") = lv47_1
            gv17: R.Tensor((128,), dtype="float32") = lv53_1
            gv18: R.Tensor((128,), dtype="float32") = lv54_1
            gv19: R.Tensor((128,), dtype="float32") = lv59_1
            gv20: R.Tensor((128,), dtype="float32") = lv60_1
            gv21: R.Tensor((256,), dtype="float32") = lv66_1
            gv22: R.Tensor((256,), dtype="float32") = lv67_1
            gv23: R.Tensor((256,), dtype="float32") = lv72_1
            gv24: R.Tensor((256,), dtype="float32") = lv73_1
            gv25: R.Tensor((256,), dtype="float32") = lv77_1
            gv26: R.Tensor((256,), dtype="float32") = lv78_1
            gv27: R.Tensor((256,), dtype="float32") = lv84_1
            gv28: R.Tensor((256,), dtype="float32") = lv85_1
            gv29: R.Tensor((256,), dtype="float32") = lv90_1
            gv30: R.Tensor((256,), dtype="float32") = lv91_1
            gv31: R.Tensor((512,), dtype="float32") = lv97_1
            gv32: R.Tensor((512,), dtype="float32") = lv98_1
            gv33: R.Tensor((512,), dtype="float32") = lv103_1
            gv34: R.Tensor((512,), dtype="float32") = lv104_1
            gv35: R.Tensor((512,), dtype="float32") = lv108_1
            gv36: R.Tensor((512,), dtype="float32") = lv109_1
            gv37: R.Tensor((512,), dtype="float32") = lv115_1
            gv38: R.Tensor((512,), dtype="float32") = lv116_1
            gv39: R.Tensor((512,), dtype="float32") = lv121_1
            gv40: R.Tensor((512,), dtype="float32") = lv122_1
            lv_2 = R.call_tir(cls.log_softmax, (gv,), out_sinfo=R.Tensor((32, 10), dtype="float32"))
            gv_1 = R.call_tir(cls.nll_loss_without_weight, (lv_2, targets), out_sinfo=R.Tensor((), dtype="float32"))
            gv_adjoint = R.call_tir(cls.ones, R.tuple(), out_sinfo=R.Tensor((), dtype="float32"))
            lv_adjoint = R.call_tir(cls.te_nll_loss_backward_no_weight, (gv_adjoint, lv_2, targets), out_sinfo=R.Tensor((32, 10), dtype="float32"))
            lv_3 = R.call_tir(cls.sum, (lv_adjoint,), out_sinfo=R.Tensor((32, 1), dtype="float32"))
            lv1_2 = R.call_tir(cls.tir_exp, (lv_2,), out_sinfo=R.Tensor((32, 10), dtype="float32"))
            lv2_2 = R.call_tir(cls.multiply12, (lv_3, lv1_2), out_sinfo=R.Tensor((32, 10), dtype="float32"))
            gv_adjoint1 = R.call_tir(cls.subtract4, (lv_adjoint, lv2_2), out_sinfo=R.Tensor((32, 10), dtype="float32"))
            lv128_adjoint: R.Tensor((32, 10), dtype="float32") = gv_adjoint1
            lv127_adjoint: R.Tensor((32, 10), dtype="float32") = lv128_adjoint
            ln_bias_adjoint = R.call_tir(cls.collapse_sum, (lv128_adjoint,), out_sinfo=R.Tensor((10,), dtype="float32"))
            lv3_2 = R.call_tir(cls.transpose, (ln_weight,), out_sinfo=R.Tensor((10, 512), dtype="float32"))
            lv4_2 = R.call_tir(cls.transpose1, (lv126_1,), out_sinfo=R.Tensor((512, 32), dtype="float32"))
            lv126_adjoint = R.call_tir(cls.matmul1, (lv127_adjoint, lv3_2), out_sinfo=R.Tensor((32, 512), dtype="float32"))
            ln_weight_adjoint = R.call_tir(cls.matmul2, (lv4_2, lv127_adjoint), out_sinfo=R.Tensor((512, 10), dtype="float32"))
            lv125_adjoint = R.call_tir(cls.reshape1, (lv126_adjoint,), out_sinfo=R.Tensor((32, 512, 1, 1), dtype="float32"))
            lv124_adjoint = R.call_tir(cls.avg_pool2d_backward, (lv125_adjoint, lv124_1), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv5_2 = R.call_tir(cls.less, (lv123_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="bool"))
            lv6_2 = R.call_tir(cls.where, (lv5_2, R.const(0, "float32"), R.const(1, "float32")), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv123_adjoint = R.call_tir(cls.multiply13, (lv6_2, lv124_adjoint), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv120_adjoint: R.Tensor((32, 512, 4, 4), dtype="float32") = lv123_adjoint
            lv111_adjoint: R.Tensor((32, 512, 4, 4), dtype="float32") = lv123_adjoint
            lv7_2 = R.call_tir(cls.zeros, R.tuple(), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv8_2 = R.call_tir(cls.zeros, R.tuple(), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv119_adjoint: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv120_adjoint, lv7_2, lv8_2
            lv353_adjoint: R.Tensor((32, 512, 4, 4), dtype="float32") = lv119_adjoint[0]
            lv356_adjoint: R.Tensor((512,), dtype="float32") = lv119_adjoint[1]
            lv359_adjoint: R.Tensor((512,), dtype="float32") = lv119_adjoint[2]
            lv358_adjoint: R.Tensor((512,), dtype="float32") = lv359_adjoint
            lv345_adjoint = R.call_tir(cls.multiply14, (lv358_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv355_adjoint: R.Tensor((512,), dtype="float32") = lv356_adjoint
            lv342_adjoint = R.call_tir(cls.multiply14, (lv355_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv351_adjoint: R.Tensor((32, 512, 4, 4), dtype="float32") = lv353_adjoint
            lv352_adjoint = R.call_tir(cls.collapse_sum1, (lv353_adjoint,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            bn_beta_adjoint = R.call_tir(cls.squeeze, (lv352_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv349_adjoint = R.call_tir(cls.multiply3, (lv351_adjoint, lv350), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv13_2 = R.call_tir(cls.multiply13, (lv351_adjoint, lv349), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv350_adjoint = R.call_tir(cls.collapse_sum1, (lv13_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            bn_gamma_adjoint = R.call_tir(cls.squeeze, (lv350_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv14_2 = R.call_tir(cls.tir_negative, (lv349_adjoint,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv15_2 = R.call_tir(cls.multiply13, (lv14_2, lv349), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv344_adjoint = R.call_tir(cls.divide3, (lv349_adjoint, lv348), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv16_2 = R.call_tir(cls.divide3, (lv15_2, lv348), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv348_adjoint = R.call_tir(cls.collapse_sum1, (lv16_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv17_2 = R.call_tir(cls.multiply15, (lv348_adjoint,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv347_adjoint = R.call_tir(cls.divide4, (lv17_2, lv348), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv346_adjoint: R.Tensor((1, 512, 1, 1), dtype="float32") = lv347_adjoint
            lv18_2 = R.call_tir(cls.squeeze, (lv346_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv345_adjoint1 = R.call_tir(cls.add16, (lv345_adjoint, lv18_2), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv19_2 = R.call_tir(cls.expand_dims3, (lv345_adjoint1,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv20_2 = R.call_tir(cls.multiply16, (lv118_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv21_2 = R.call_tir(cls.collapse_sum1, (lv118_1,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv22_2 = R.call_tir(cls.multiply17, (lv21_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv23_2 = R.call_tir(cls.subtract3, (lv20_2, lv22_2), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv118_adjoint = R.call_tir(cls.multiply18, (lv19_2, lv23_2), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv118_adjoint1 = R.call_tir(cls.add11, (lv118_adjoint, lv344_adjoint), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv24_2 = R.call_tir(cls.tir_negative, (lv344_adjoint,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv343_adjoint = R.call_tir(cls.collapse_sum1, (lv24_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv25_2 = R.call_tir(cls.squeeze, (lv343_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv342_adjoint1 = R.call_tir(cls.add16, (lv342_adjoint, lv25_2), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv26_2 = R.call_tir(cls.divide5, (lv342_adjoint1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv27_2 = R.call_tir(cls.expand_dims3, (lv26_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv28_2 = R.call_tir(cls.broadcast_to, (lv27_2,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv118_adjoint2 = R.call_tir(cls.add11, (lv118_adjoint1, lv28_2), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv117_adjoint = R.call_tir(cls.conv2d_transpose, (lv118_adjoint2, conv2d_weight_19), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            conv2d_weight_adjoint = R.call_tir(cls.conv2d11, (lv117_1, lv118_adjoint2), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            lv29_2 = R.call_tir(cls.less, (lv114_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="bool"))
            lv30_2 = R.call_tir(cls.where, (lv29_2, R.const(0, "float32"), R.const(1, "float32")), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv114_adjoint = R.call_tir(cls.multiply13, (lv30_2, lv117_adjoint), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv31_2 = R.call_tir(cls.zeros, R.tuple(), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv32_2 = R.call_tir(cls.zeros, R.tuple(), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv113_adjoint: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv114_adjoint, lv31_2, lv32_2
            lv335_adjoint: R.Tensor((32, 512, 4, 4), dtype="float32") = lv113_adjoint[0]
            lv338_adjoint: R.Tensor((512,), dtype="float32") = lv113_adjoint[1]
            lv341_adjoint: R.Tensor((512,), dtype="float32") = lv113_adjoint[2]
            lv340_adjoint: R.Tensor((512,), dtype="float32") = lv341_adjoint
            lv327_adjoint = R.call_tir(cls.multiply14, (lv340_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv337_adjoint: R.Tensor((512,), dtype="float32") = lv338_adjoint
            lv324_adjoint = R.call_tir(cls.multiply14, (lv337_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv333_adjoint: R.Tensor((32, 512, 4, 4), dtype="float32") = lv335_adjoint
            lv334_adjoint = R.call_tir(cls.collapse_sum1, (lv335_adjoint,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            bn_beta_adjoint1 = R.call_tir(cls.squeeze, (lv334_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv331_adjoint = R.call_tir(cls.multiply3, (lv333_adjoint, lv332), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv37_2 = R.call_tir(cls.multiply13, (lv333_adjoint, lv331), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv332_adjoint = R.call_tir(cls.collapse_sum1, (lv37_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            bn_gamma_adjoint1 = R.call_tir(cls.squeeze, (lv332_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv38_2 = R.call_tir(cls.tir_negative, (lv331_adjoint,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv39_2 = R.call_tir(cls.multiply13, (lv38_2, lv331), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv326_adjoint = R.call_tir(cls.divide3, (lv331_adjoint, lv330), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv40_2 = R.call_tir(cls.divide3, (lv39_2, lv330), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv330_adjoint = R.call_tir(cls.collapse_sum1, (lv40_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv41_2 = R.call_tir(cls.multiply15, (lv330_adjoint,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv329_adjoint = R.call_tir(cls.divide4, (lv41_2, lv330), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv328_adjoint: R.Tensor((1, 512, 1, 1), dtype="float32") = lv329_adjoint
            lv42_2 = R.call_tir(cls.squeeze, (lv328_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv327_adjoint1 = R.call_tir(cls.add16, (lv327_adjoint, lv42_2), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv43_2 = R.call_tir(cls.expand_dims3, (lv327_adjoint1,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv44_2 = R.call_tir(cls.multiply16, (lv112_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv45_2 = R.call_tir(cls.collapse_sum1, (lv112_1,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv46_2 = R.call_tir(cls.multiply17, (lv45_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv47_2 = R.call_tir(cls.subtract3, (lv44_2, lv46_2), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv112_adjoint = R.call_tir(cls.multiply18, (lv43_2, lv47_2), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv112_adjoint1 = R.call_tir(cls.add11, (lv112_adjoint, lv326_adjoint), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv48_2 = R.call_tir(cls.tir_negative, (lv326_adjoint,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv325_adjoint = R.call_tir(cls.collapse_sum1, (lv48_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv49_2 = R.call_tir(cls.squeeze, (lv325_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv324_adjoint1 = R.call_tir(cls.add16, (lv324_adjoint, lv49_2), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv50_2 = R.call_tir(cls.divide5, (lv324_adjoint1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv51_2 = R.call_tir(cls.expand_dims3, (lv50_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv52_2 = R.call_tir(cls.broadcast_to, (lv51_2,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv112_adjoint2 = R.call_tir(cls.add11, (lv112_adjoint1, lv52_2), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv53_2 = R.call_tir(cls.conv2d_transpose, (lv112_adjoint2, conv2d_weight_18), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv111_adjoint1 = R.call_tir(cls.add11, (lv111_adjoint, lv53_2), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            conv2d_weight_adjoint1 = R.call_tir(cls.conv2d11, (lv111_1, lv112_adjoint2), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            lv54_2 = R.call_tir(cls.less, (lv110_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="bool"))
            lv55_2 = R.call_tir(cls.where, (lv54_2, R.const(0, "float32"), R.const(1, "float32")), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv110_adjoint = R.call_tir(cls.multiply13, (lv55_2, lv111_adjoint1), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv102_adjoint: R.Tensor((32, 512, 4, 4), dtype="float32") = lv110_adjoint
            lv107_adjoint: R.Tensor((32, 512, 4, 4), dtype="float32") = lv110_adjoint
            lv56_2 = R.call_tir(cls.zeros, R.tuple(), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv57_2 = R.call_tir(cls.zeros, R.tuple(), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv106_adjoint: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv107_adjoint, lv56_2, lv57_2
            lv317_adjoint: R.Tensor((32, 512, 4, 4), dtype="float32") = lv106_adjoint[0]
            lv320_adjoint: R.Tensor((512,), dtype="float32") = lv106_adjoint[1]
            lv323_adjoint: R.Tensor((512,), dtype="float32") = lv106_adjoint[2]
            lv322_adjoint: R.Tensor((512,), dtype="float32") = lv323_adjoint
            lv309_adjoint = R.call_tir(cls.multiply14, (lv322_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv319_adjoint: R.Tensor((512,), dtype="float32") = lv320_adjoint
            lv306_adjoint = R.call_tir(cls.multiply14, (lv319_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv315_adjoint: R.Tensor((32, 512, 4, 4), dtype="float32") = lv317_adjoint
            lv316_adjoint = R.call_tir(cls.collapse_sum1, (lv317_adjoint,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            bn_beta_adjoint2 = R.call_tir(cls.squeeze, (lv316_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv313_adjoint = R.call_tir(cls.multiply3, (lv315_adjoint, lv314), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv62_2 = R.call_tir(cls.multiply13, (lv315_adjoint, lv313), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv314_adjoint = R.call_tir(cls.collapse_sum1, (lv62_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            bn_gamma_adjoint2 = R.call_tir(cls.squeeze, (lv314_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv63_2 = R.call_tir(cls.tir_negative, (lv313_adjoint,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv64_2 = R.call_tir(cls.multiply13, (lv63_2, lv313), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv308_adjoint = R.call_tir(cls.divide3, (lv313_adjoint, lv312), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv65_2 = R.call_tir(cls.divide3, (lv64_2, lv312), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv312_adjoint = R.call_tir(cls.collapse_sum1, (lv65_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv66_2 = R.call_tir(cls.multiply15, (lv312_adjoint,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv311_adjoint = R.call_tir(cls.divide4, (lv66_2, lv312), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv310_adjoint: R.Tensor((1, 512, 1, 1), dtype="float32") = lv311_adjoint
            lv67_2 = R.call_tir(cls.squeeze, (lv310_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv309_adjoint1 = R.call_tir(cls.add16, (lv309_adjoint, lv67_2), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv68_2 = R.call_tir(cls.expand_dims3, (lv309_adjoint1,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv69_2 = R.call_tir(cls.multiply16, (lv105_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv70_2 = R.call_tir(cls.collapse_sum1, (lv105_1,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv71_2 = R.call_tir(cls.multiply17, (lv70_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv72_2 = R.call_tir(cls.subtract3, (lv69_2, lv71_2), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv105_adjoint = R.call_tir(cls.multiply18, (lv68_2, lv72_2), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv105_adjoint1 = R.call_tir(cls.add11, (lv105_adjoint, lv308_adjoint), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv73_2 = R.call_tir(cls.tir_negative, (lv308_adjoint,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv307_adjoint = R.call_tir(cls.collapse_sum1, (lv73_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv74_2 = R.call_tir(cls.squeeze, (lv307_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv306_adjoint1 = R.call_tir(cls.add16, (lv306_adjoint, lv74_2), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv75_2 = R.call_tir(cls.divide5, (lv306_adjoint1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv76_2 = R.call_tir(cls.expand_dims3, (lv75_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv77_2 = R.call_tir(cls.broadcast_to, (lv76_2,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv105_adjoint2 = R.call_tir(cls.add11, (lv105_adjoint1, lv77_2), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv93_adjoint = R.call_tir(cls.conv2d_transpose1, (lv105_adjoint2, conv2d_weight_17), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv78_2 = R.call_tir(cls.conv2d12, (lv93_1, lv105_adjoint2), out_sinfo=R.Tensor((512, 256, 2, 2), dtype="float32"))
            conv2d_weight_adjoint2 = R.call_tir(cls.strided_slice, (lv78_2,), out_sinfo=R.Tensor((512, 256, 1, 1), dtype="float32"))
            lv79_2 = R.call_tir(cls.zeros, R.tuple(), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv80_2 = R.call_tir(cls.zeros, R.tuple(), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv101_adjoint: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv102_adjoint, lv79_2, lv80_2
            lv299_adjoint: R.Tensor((32, 512, 4, 4), dtype="float32") = lv101_adjoint[0]
            lv302_adjoint: R.Tensor((512,), dtype="float32") = lv101_adjoint[1]
            lv305_adjoint: R.Tensor((512,), dtype="float32") = lv101_adjoint[2]
            lv304_adjoint: R.Tensor((512,), dtype="float32") = lv305_adjoint
            lv291_adjoint = R.call_tir(cls.multiply14, (lv304_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv301_adjoint: R.Tensor((512,), dtype="float32") = lv302_adjoint
            lv288_adjoint = R.call_tir(cls.multiply14, (lv301_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv297_adjoint: R.Tensor((32, 512, 4, 4), dtype="float32") = lv299_adjoint
            lv298_adjoint = R.call_tir(cls.collapse_sum1, (lv299_adjoint,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            bn_beta_adjoint3 = R.call_tir(cls.squeeze, (lv298_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv295_adjoint = R.call_tir(cls.multiply3, (lv297_adjoint, lv296), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv85_2 = R.call_tir(cls.multiply13, (lv297_adjoint, lv295), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv296_adjoint = R.call_tir(cls.collapse_sum1, (lv85_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            bn_gamma_adjoint3 = R.call_tir(cls.squeeze, (lv296_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv86_2 = R.call_tir(cls.tir_negative, (lv295_adjoint,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv87_2 = R.call_tir(cls.multiply13, (lv86_2, lv295), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv290_adjoint = R.call_tir(cls.divide3, (lv295_adjoint, lv294), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv88_2 = R.call_tir(cls.divide3, (lv87_2, lv294), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv294_adjoint = R.call_tir(cls.collapse_sum1, (lv88_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv89_2 = R.call_tir(cls.multiply15, (lv294_adjoint,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv293_adjoint = R.call_tir(cls.divide4, (lv89_2, lv294), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv292_adjoint: R.Tensor((1, 512, 1, 1), dtype="float32") = lv293_adjoint
            lv90_2 = R.call_tir(cls.squeeze, (lv292_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv291_adjoint1 = R.call_tir(cls.add16, (lv291_adjoint, lv90_2), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv91_2 = R.call_tir(cls.expand_dims3, (lv291_adjoint1,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv92_2 = R.call_tir(cls.multiply16, (lv100_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv93_2 = R.call_tir(cls.collapse_sum1, (lv100_1,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv94_2 = R.call_tir(cls.multiply17, (lv93_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv95_2 = R.call_tir(cls.subtract3, (lv92_2, lv94_2), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv100_adjoint = R.call_tir(cls.multiply18, (lv91_2, lv95_2), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv100_adjoint1 = R.call_tir(cls.add11, (lv100_adjoint, lv290_adjoint), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv96_2 = R.call_tir(cls.tir_negative, (lv290_adjoint,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv289_adjoint = R.call_tir(cls.collapse_sum1, (lv96_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv97_2 = R.call_tir(cls.squeeze, (lv289_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv288_adjoint1 = R.call_tir(cls.add16, (lv288_adjoint, lv97_2), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv98_2 = R.call_tir(cls.divide5, (lv288_adjoint1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv99_2 = R.call_tir(cls.expand_dims3, (lv98_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv100_2 = R.call_tir(cls.broadcast_to, (lv99_2,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv100_adjoint2 = R.call_tir(cls.add11, (lv100_adjoint1, lv100_2), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv99_adjoint = R.call_tir(cls.conv2d_transpose, (lv100_adjoint2, conv2d_weight_16), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            conv2d_weight_adjoint3 = R.call_tir(cls.conv2d11, (lv99_1, lv100_adjoint2), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            lv101_2 = R.call_tir(cls.less, (lv96_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="bool"))
            lv102_2 = R.call_tir(cls.where, (lv101_2, R.const(0, "float32"), R.const(1, "float32")), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv96_adjoint = R.call_tir(cls.multiply13, (lv102_2, lv99_adjoint), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv103_2 = R.call_tir(cls.zeros, R.tuple(), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv104_2 = R.call_tir(cls.zeros, R.tuple(), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv95_adjoint: R.Tuple(R.Tensor((32, 512, 4, 4), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = lv96_adjoint, lv103_2, lv104_2
            lv281_adjoint: R.Tensor((32, 512, 4, 4), dtype="float32") = lv95_adjoint[0]
            lv284_adjoint: R.Tensor((512,), dtype="float32") = lv95_adjoint[1]
            lv287_adjoint: R.Tensor((512,), dtype="float32") = lv95_adjoint[2]
            lv286_adjoint: R.Tensor((512,), dtype="float32") = lv287_adjoint
            lv273_adjoint = R.call_tir(cls.multiply14, (lv286_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv283_adjoint: R.Tensor((512,), dtype="float32") = lv284_adjoint
            lv270_adjoint = R.call_tir(cls.multiply14, (lv283_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv279_adjoint: R.Tensor((32, 512, 4, 4), dtype="float32") = lv281_adjoint
            lv280_adjoint = R.call_tir(cls.collapse_sum1, (lv281_adjoint,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            bn_beta_adjoint4 = R.call_tir(cls.squeeze, (lv280_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv277_adjoint = R.call_tir(cls.multiply3, (lv279_adjoint, lv278), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv109_2 = R.call_tir(cls.multiply13, (lv279_adjoint, lv277), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv278_adjoint = R.call_tir(cls.collapse_sum1, (lv109_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            bn_gamma_adjoint4 = R.call_tir(cls.squeeze, (lv278_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv110_2 = R.call_tir(cls.tir_negative, (lv277_adjoint,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv111_2 = R.call_tir(cls.multiply13, (lv110_2, lv277), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv272_adjoint = R.call_tir(cls.divide3, (lv277_adjoint, lv276), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv112_2 = R.call_tir(cls.divide3, (lv111_2, lv276), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv276_adjoint = R.call_tir(cls.collapse_sum1, (lv112_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv113_2 = R.call_tir(cls.multiply15, (lv276_adjoint,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv275_adjoint = R.call_tir(cls.divide4, (lv113_2, lv276), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv274_adjoint: R.Tensor((1, 512, 1, 1), dtype="float32") = lv275_adjoint
            lv114_2 = R.call_tir(cls.squeeze, (lv274_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv273_adjoint1 = R.call_tir(cls.add16, (lv273_adjoint, lv114_2), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv115_2 = R.call_tir(cls.expand_dims3, (lv273_adjoint1,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv116_2 = R.call_tir(cls.multiply16, (lv94_1,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv117_2 = R.call_tir(cls.collapse_sum1, (lv94_1,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv118_2 = R.call_tir(cls.multiply17, (lv117_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv119_2 = R.call_tir(cls.subtract3, (lv116_2, lv118_2), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv94_adjoint = R.call_tir(cls.multiply18, (lv115_2, lv119_2), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv94_adjoint1 = R.call_tir(cls.add11, (lv94_adjoint, lv272_adjoint), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv120_2 = R.call_tir(cls.tir_negative, (lv272_adjoint,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv271_adjoint = R.call_tir(cls.collapse_sum1, (lv120_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv121_2 = R.call_tir(cls.squeeze, (lv271_adjoint,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv270_adjoint1 = R.call_tir(cls.add16, (lv270_adjoint, lv121_2), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv122_2 = R.call_tir(cls.divide5, (lv270_adjoint1,), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv123_2 = R.call_tir(cls.expand_dims3, (lv122_2,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv124_2 = R.call_tir(cls.broadcast_to, (lv123_2,), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv94_adjoint2 = R.call_tir(cls.add11, (lv94_adjoint1, lv124_2), out_sinfo=R.Tensor((32, 512, 4, 4), dtype="float32"))
            lv125_2 = R.call_tir(cls.conv2d_transpose2, (lv94_adjoint2, conv2d_weight_15), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv93_adjoint1 = R.call_tir(cls.add8, (lv93_adjoint, lv125_2), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv126_2 = R.call_tir(cls.conv2d13, (lv93_1, lv94_adjoint2), out_sinfo=R.Tensor((512, 256, 4, 4), dtype="float32"))
            conv2d_weight_adjoint4 = R.call_tir(cls.strided_slice1, (lv126_2,), out_sinfo=R.Tensor((512, 256, 3, 3), dtype="float32"))
            lv127_2 = R.call_tir(cls.less1, (lv92_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="bool"))
            lv128_2 = R.call_tir(cls.where1, (lv127_2, R.const(0, "float32"), R.const(1, "float32")), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv92_adjoint = R.call_tir(cls.multiply19, (lv128_2, lv93_adjoint1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv89_adjoint: R.Tensor((32, 256, 8, 8), dtype="float32") = lv92_adjoint
            lv80_adjoint: R.Tensor((32, 256, 8, 8), dtype="float32") = lv92_adjoint
            lv129_1 = R.call_tir(cls.zeros1, R.tuple(), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv130_1 = R.call_tir(cls.zeros1, R.tuple(), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv88_adjoint: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv89_adjoint, lv129_1, lv130_1
            lv263_adjoint: R.Tensor((32, 256, 8, 8), dtype="float32") = lv88_adjoint[0]
            lv266_adjoint: R.Tensor((256,), dtype="float32") = lv88_adjoint[1]
            lv269_adjoint: R.Tensor((256,), dtype="float32") = lv88_adjoint[2]
            lv268_adjoint: R.Tensor((256,), dtype="float32") = lv269_adjoint
            lv255_adjoint = R.call_tir(cls.multiply20, (lv268_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv265_adjoint: R.Tensor((256,), dtype="float32") = lv266_adjoint
            lv252_adjoint = R.call_tir(cls.multiply20, (lv265_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv261_adjoint: R.Tensor((32, 256, 8, 8), dtype="float32") = lv263_adjoint
            lv262_adjoint = R.call_tir(cls.collapse_sum2, (lv263_adjoint,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            bn_beta_adjoint5 = R.call_tir(cls.squeeze1, (lv262_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv259_adjoint = R.call_tir(cls.multiply2, (lv261_adjoint, lv260), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv135_1 = R.call_tir(cls.multiply19, (lv261_adjoint, lv259), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv260_adjoint = R.call_tir(cls.collapse_sum2, (lv135_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            bn_gamma_adjoint5 = R.call_tir(cls.squeeze1, (lv260_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv136_1 = R.call_tir(cls.tir_negative1, (lv259_adjoint,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv137_1 = R.call_tir(cls.multiply19, (lv136_1, lv259), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv254_adjoint = R.call_tir(cls.divide2, (lv259_adjoint, lv258), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv138_1 = R.call_tir(cls.divide2, (lv137_1, lv258), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv258_adjoint = R.call_tir(cls.collapse_sum2, (lv138_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv139_1 = R.call_tir(cls.multiply21, (lv258_adjoint,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv257_adjoint = R.call_tir(cls.divide6, (lv139_1, lv258), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv256_adjoint: R.Tensor((1, 256, 1, 1), dtype="float32") = lv257_adjoint
            lv140_1 = R.call_tir(cls.squeeze1, (lv256_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv255_adjoint1 = R.call_tir(cls.add15, (lv255_adjoint, lv140_1), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv141_1 = R.call_tir(cls.expand_dims2, (lv255_adjoint1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv142_1 = R.call_tir(cls.multiply22, (lv87_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv143_1 = R.call_tir(cls.collapse_sum2, (lv87_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv144_1 = R.call_tir(cls.multiply23, (lv143_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv145_1 = R.call_tir(cls.subtract2, (lv142_1, lv144_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv87_adjoint = R.call_tir(cls.multiply24, (lv141_1, lv145_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv87_adjoint1 = R.call_tir(cls.add8, (lv87_adjoint, lv254_adjoint), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv146_1 = R.call_tir(cls.tir_negative1, (lv254_adjoint,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv253_adjoint = R.call_tir(cls.collapse_sum2, (lv146_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv147_1 = R.call_tir(cls.squeeze1, (lv253_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv252_adjoint1 = R.call_tir(cls.add15, (lv252_adjoint, lv147_1), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv148_1 = R.call_tir(cls.divide7, (lv252_adjoint1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv149_1 = R.call_tir(cls.expand_dims2, (lv148_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv150_1 = R.call_tir(cls.broadcast_to1, (lv149_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv87_adjoint2 = R.call_tir(cls.add8, (lv87_adjoint1, lv150_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv86_adjoint = R.call_tir(cls.conv2d_transpose3, (lv87_adjoint2, conv2d_weight_14), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            conv2d_weight_adjoint5 = R.call_tir(cls.conv2d14, (lv86_1, lv87_adjoint2), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            lv151_1 = R.call_tir(cls.less1, (lv83_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="bool"))
            lv152_1 = R.call_tir(cls.where1, (lv151_1, R.const(0, "float32"), R.const(1, "float32")), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv83_adjoint = R.call_tir(cls.multiply19, (lv152_1, lv86_adjoint), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv153_1 = R.call_tir(cls.zeros1, R.tuple(), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv154_1 = R.call_tir(cls.zeros1, R.tuple(), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv82_adjoint: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv83_adjoint, lv153_1, lv154_1
            lv245_adjoint: R.Tensor((32, 256, 8, 8), dtype="float32") = lv82_adjoint[0]
            lv248_adjoint: R.Tensor((256,), dtype="float32") = lv82_adjoint[1]
            lv251_adjoint: R.Tensor((256,), dtype="float32") = lv82_adjoint[2]
            lv250_adjoint: R.Tensor((256,), dtype="float32") = lv251_adjoint
            lv237_adjoint = R.call_tir(cls.multiply20, (lv250_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv247_adjoint: R.Tensor((256,), dtype="float32") = lv248_adjoint
            lv234_adjoint = R.call_tir(cls.multiply20, (lv247_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv243_adjoint: R.Tensor((32, 256, 8, 8), dtype="float32") = lv245_adjoint
            lv244_adjoint = R.call_tir(cls.collapse_sum2, (lv245_adjoint,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            bn_beta_adjoint6 = R.call_tir(cls.squeeze1, (lv244_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv241_adjoint = R.call_tir(cls.multiply2, (lv243_adjoint, lv242), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv159_1 = R.call_tir(cls.multiply19, (lv243_adjoint, lv241), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv242_adjoint = R.call_tir(cls.collapse_sum2, (lv159_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            bn_gamma_adjoint6 = R.call_tir(cls.squeeze1, (lv242_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv160_1 = R.call_tir(cls.tir_negative1, (lv241_adjoint,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv161_1 = R.call_tir(cls.multiply19, (lv160_1, lv241), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv236_adjoint = R.call_tir(cls.divide2, (lv241_adjoint, lv240), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv162_1 = R.call_tir(cls.divide2, (lv161_1, lv240), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv240_adjoint = R.call_tir(cls.collapse_sum2, (lv162_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv163_1 = R.call_tir(cls.multiply21, (lv240_adjoint,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv239_adjoint = R.call_tir(cls.divide6, (lv163_1, lv240), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv238_adjoint: R.Tensor((1, 256, 1, 1), dtype="float32") = lv239_adjoint
            lv164_1 = R.call_tir(cls.squeeze1, (lv238_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv237_adjoint1 = R.call_tir(cls.add15, (lv237_adjoint, lv164_1), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv165_1 = R.call_tir(cls.expand_dims2, (lv237_adjoint1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv166_1 = R.call_tir(cls.multiply22, (lv81_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv167_1 = R.call_tir(cls.collapse_sum2, (lv81_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv168_1 = R.call_tir(cls.multiply23, (lv167_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv169_1 = R.call_tir(cls.subtract2, (lv166_1, lv168_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv81_adjoint = R.call_tir(cls.multiply24, (lv165_1, lv169_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv81_adjoint1 = R.call_tir(cls.add8, (lv81_adjoint, lv236_adjoint), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv170_1 = R.call_tir(cls.tir_negative1, (lv236_adjoint,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv235_adjoint = R.call_tir(cls.collapse_sum2, (lv170_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv171_1 = R.call_tir(cls.squeeze1, (lv235_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv234_adjoint1 = R.call_tir(cls.add15, (lv234_adjoint, lv171_1), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv172_1 = R.call_tir(cls.divide7, (lv234_adjoint1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv173_1 = R.call_tir(cls.expand_dims2, (lv172_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv174_1 = R.call_tir(cls.broadcast_to1, (lv173_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv81_adjoint2 = R.call_tir(cls.add8, (lv81_adjoint1, lv174_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv175_1 = R.call_tir(cls.conv2d_transpose3, (lv81_adjoint2, conv2d_weight_13), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv80_adjoint1 = R.call_tir(cls.add8, (lv80_adjoint, lv175_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            conv2d_weight_adjoint6 = R.call_tir(cls.conv2d14, (lv80_1, lv81_adjoint2), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            lv176_1 = R.call_tir(cls.less1, (lv79_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="bool"))
            lv177_1 = R.call_tir(cls.where1, (lv176_1, R.const(0, "float32"), R.const(1, "float32")), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv79_adjoint = R.call_tir(cls.multiply19, (lv177_1, lv80_adjoint1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv71_adjoint: R.Tensor((32, 256, 8, 8), dtype="float32") = lv79_adjoint
            lv76_adjoint: R.Tensor((32, 256, 8, 8), dtype="float32") = lv79_adjoint
            lv178_1 = R.call_tir(cls.zeros1, R.tuple(), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv179_1 = R.call_tir(cls.zeros1, R.tuple(), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv75_adjoint: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv76_adjoint, lv178_1, lv179_1
            lv227_adjoint: R.Tensor((32, 256, 8, 8), dtype="float32") = lv75_adjoint[0]
            lv230_adjoint: R.Tensor((256,), dtype="float32") = lv75_adjoint[1]
            lv233_adjoint: R.Tensor((256,), dtype="float32") = lv75_adjoint[2]
            lv232_adjoint: R.Tensor((256,), dtype="float32") = lv233_adjoint
            lv219_adjoint = R.call_tir(cls.multiply20, (lv232_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv229_adjoint: R.Tensor((256,), dtype="float32") = lv230_adjoint
            lv216_adjoint = R.call_tir(cls.multiply20, (lv229_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv225_adjoint: R.Tensor((32, 256, 8, 8), dtype="float32") = lv227_adjoint
            lv226_adjoint = R.call_tir(cls.collapse_sum2, (lv227_adjoint,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            bn_beta_adjoint7 = R.call_tir(cls.squeeze1, (lv226_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv223_adjoint = R.call_tir(cls.multiply2, (lv225_adjoint, lv224), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv184_1 = R.call_tir(cls.multiply19, (lv225_adjoint, lv223), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv224_adjoint = R.call_tir(cls.collapse_sum2, (lv184_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            bn_gamma_adjoint7 = R.call_tir(cls.squeeze1, (lv224_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv185_1 = R.call_tir(cls.tir_negative1, (lv223_adjoint,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv186_1 = R.call_tir(cls.multiply19, (lv185_1, lv223), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv218_adjoint = R.call_tir(cls.divide2, (lv223_adjoint, lv222), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv187_1 = R.call_tir(cls.divide2, (lv186_1, lv222), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv222_adjoint = R.call_tir(cls.collapse_sum2, (lv187_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv188_1 = R.call_tir(cls.multiply21, (lv222_adjoint,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv221_adjoint = R.call_tir(cls.divide6, (lv188_1, lv222), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv220_adjoint: R.Tensor((1, 256, 1, 1), dtype="float32") = lv221_adjoint
            lv189_1 = R.call_tir(cls.squeeze1, (lv220_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv219_adjoint1 = R.call_tir(cls.add15, (lv219_adjoint, lv189_1), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv190_1 = R.call_tir(cls.expand_dims2, (lv219_adjoint1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv191_1 = R.call_tir(cls.multiply22, (lv74_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv192_1 = R.call_tir(cls.collapse_sum2, (lv74_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv193_1 = R.call_tir(cls.multiply23, (lv192_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv194_1 = R.call_tir(cls.subtract2, (lv191_1, lv193_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv74_adjoint = R.call_tir(cls.multiply24, (lv190_1, lv194_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv74_adjoint1 = R.call_tir(cls.add8, (lv74_adjoint, lv218_adjoint), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv195_1 = R.call_tir(cls.tir_negative1, (lv218_adjoint,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv217_adjoint = R.call_tir(cls.collapse_sum2, (lv195_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv196_1 = R.call_tir(cls.squeeze1, (lv217_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv216_adjoint1 = R.call_tir(cls.add15, (lv216_adjoint, lv196_1), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv197_1 = R.call_tir(cls.divide7, (lv216_adjoint1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv198_1 = R.call_tir(cls.expand_dims2, (lv197_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv199_1 = R.call_tir(cls.broadcast_to1, (lv198_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv74_adjoint2 = R.call_tir(cls.add8, (lv74_adjoint1, lv199_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv62_adjoint = R.call_tir(cls.conv2d_transpose4, (lv74_adjoint2, conv2d_weight_12), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv200_1 = R.call_tir(cls.conv2d15, (lv62_1, lv74_adjoint2), out_sinfo=R.Tensor((256, 128, 2, 2), dtype="float32"))
            conv2d_weight_adjoint7 = R.call_tir(cls.strided_slice2, (lv200_1,), out_sinfo=R.Tensor((256, 128, 1, 1), dtype="float32"))
            lv201_1 = R.call_tir(cls.zeros1, R.tuple(), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv202_1 = R.call_tir(cls.zeros1, R.tuple(), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv70_adjoint: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv71_adjoint, lv201_1, lv202_1
            lv209_adjoint: R.Tensor((32, 256, 8, 8), dtype="float32") = lv70_adjoint[0]
            lv212_adjoint: R.Tensor((256,), dtype="float32") = lv70_adjoint[1]
            lv215_adjoint: R.Tensor((256,), dtype="float32") = lv70_adjoint[2]
            lv214_adjoint: R.Tensor((256,), dtype="float32") = lv215_adjoint
            lv201_adjoint = R.call_tir(cls.multiply20, (lv214_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv211_adjoint: R.Tensor((256,), dtype="float32") = lv212_adjoint
            lv198_adjoint = R.call_tir(cls.multiply20, (lv211_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv207_adjoint: R.Tensor((32, 256, 8, 8), dtype="float32") = lv209_adjoint
            lv208_adjoint = R.call_tir(cls.collapse_sum2, (lv209_adjoint,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            bn_beta_adjoint8 = R.call_tir(cls.squeeze1, (lv208_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv205_adjoint = R.call_tir(cls.multiply2, (lv207_adjoint, lv206), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv207_1 = R.call_tir(cls.multiply19, (lv207_adjoint, lv205), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv206_adjoint = R.call_tir(cls.collapse_sum2, (lv207_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            bn_gamma_adjoint8 = R.call_tir(cls.squeeze1, (lv206_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv208_1 = R.call_tir(cls.tir_negative1, (lv205_adjoint,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv209_1 = R.call_tir(cls.multiply19, (lv208_1, lv205), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv200_adjoint = R.call_tir(cls.divide2, (lv205_adjoint, lv204), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv210_1 = R.call_tir(cls.divide2, (lv209_1, lv204), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv204_adjoint = R.call_tir(cls.collapse_sum2, (lv210_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv211_1 = R.call_tir(cls.multiply21, (lv204_adjoint,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv203_adjoint = R.call_tir(cls.divide6, (lv211_1, lv204), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv202_adjoint: R.Tensor((1, 256, 1, 1), dtype="float32") = lv203_adjoint
            lv212_1 = R.call_tir(cls.squeeze1, (lv202_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv201_adjoint1 = R.call_tir(cls.add15, (lv201_adjoint, lv212_1), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv213_1 = R.call_tir(cls.expand_dims2, (lv201_adjoint1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv214_1 = R.call_tir(cls.multiply22, (lv69_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv215_1 = R.call_tir(cls.collapse_sum2, (lv69_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv216_1 = R.call_tir(cls.multiply23, (lv215_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv217_1 = R.call_tir(cls.subtract2, (lv214_1, lv216_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv69_adjoint = R.call_tir(cls.multiply24, (lv213_1, lv217_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv69_adjoint1 = R.call_tir(cls.add8, (lv69_adjoint, lv200_adjoint), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv218_1 = R.call_tir(cls.tir_negative1, (lv200_adjoint,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv199_adjoint = R.call_tir(cls.collapse_sum2, (lv218_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv219_1 = R.call_tir(cls.squeeze1, (lv199_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv198_adjoint1 = R.call_tir(cls.add15, (lv198_adjoint, lv219_1), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv220_1 = R.call_tir(cls.divide7, (lv198_adjoint1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv221_1 = R.call_tir(cls.expand_dims2, (lv220_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv222_1 = R.call_tir(cls.broadcast_to1, (lv221_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv69_adjoint2 = R.call_tir(cls.add8, (lv69_adjoint1, lv222_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv68_adjoint = R.call_tir(cls.conv2d_transpose3, (lv69_adjoint2, conv2d_weight_11), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            conv2d_weight_adjoint8 = R.call_tir(cls.conv2d14, (lv68_1, lv69_adjoint2), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            lv223_1 = R.call_tir(cls.less1, (lv65_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="bool"))
            lv224_1 = R.call_tir(cls.where1, (lv223_1, R.const(0, "float32"), R.const(1, "float32")), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv65_adjoint = R.call_tir(cls.multiply19, (lv224_1, lv68_adjoint), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv225_1 = R.call_tir(cls.zeros1, R.tuple(), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv226_1 = R.call_tir(cls.zeros1, R.tuple(), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv64_adjoint: R.Tuple(R.Tensor((32, 256, 8, 8), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = lv65_adjoint, lv225_1, lv226_1
            lv191_adjoint: R.Tensor((32, 256, 8, 8), dtype="float32") = lv64_adjoint[0]
            lv194_adjoint: R.Tensor((256,), dtype="float32") = lv64_adjoint[1]
            lv197_adjoint: R.Tensor((256,), dtype="float32") = lv64_adjoint[2]
            lv196_adjoint: R.Tensor((256,), dtype="float32") = lv197_adjoint
            lv183_adjoint = R.call_tir(cls.multiply20, (lv196_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv193_adjoint: R.Tensor((256,), dtype="float32") = lv194_adjoint
            lv180_adjoint = R.call_tir(cls.multiply20, (lv193_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv189_adjoint: R.Tensor((32, 256, 8, 8), dtype="float32") = lv191_adjoint
            lv190_adjoint = R.call_tir(cls.collapse_sum2, (lv191_adjoint,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            bn_beta_adjoint9 = R.call_tir(cls.squeeze1, (lv190_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv187_adjoint = R.call_tir(cls.multiply2, (lv189_adjoint, lv188), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv231_1 = R.call_tir(cls.multiply19, (lv189_adjoint, lv187), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv188_adjoint = R.call_tir(cls.collapse_sum2, (lv231_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            bn_gamma_adjoint9 = R.call_tir(cls.squeeze1, (lv188_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv232_1 = R.call_tir(cls.tir_negative1, (lv187_adjoint,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv233_1 = R.call_tir(cls.multiply19, (lv232_1, lv187), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv182_adjoint = R.call_tir(cls.divide2, (lv187_adjoint, lv186), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv234_1 = R.call_tir(cls.divide2, (lv233_1, lv186), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv186_adjoint = R.call_tir(cls.collapse_sum2, (lv234_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv235_1 = R.call_tir(cls.multiply21, (lv186_adjoint,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv185_adjoint = R.call_tir(cls.divide6, (lv235_1, lv186), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv184_adjoint: R.Tensor((1, 256, 1, 1), dtype="float32") = lv185_adjoint
            lv236_1 = R.call_tir(cls.squeeze1, (lv184_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv183_adjoint1 = R.call_tir(cls.add15, (lv183_adjoint, lv236_1), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv237_1 = R.call_tir(cls.expand_dims2, (lv183_adjoint1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv238_1 = R.call_tir(cls.multiply22, (lv63_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv239_1 = R.call_tir(cls.collapse_sum2, (lv63_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv240_1 = R.call_tir(cls.multiply23, (lv239_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv241_1 = R.call_tir(cls.subtract2, (lv238_1, lv240_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv63_adjoint = R.call_tir(cls.multiply24, (lv237_1, lv241_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv63_adjoint1 = R.call_tir(cls.add8, (lv63_adjoint, lv182_adjoint), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv242_1 = R.call_tir(cls.tir_negative1, (lv182_adjoint,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv181_adjoint = R.call_tir(cls.collapse_sum2, (lv242_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv243_1 = R.call_tir(cls.squeeze1, (lv181_adjoint,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv180_adjoint1 = R.call_tir(cls.add15, (lv180_adjoint, lv243_1), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv244_1 = R.call_tir(cls.divide7, (lv180_adjoint1,), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv245_1 = R.call_tir(cls.expand_dims2, (lv244_1,), out_sinfo=R.Tensor((1, 256, 1, 1), dtype="float32"))
            lv246_1 = R.call_tir(cls.broadcast_to1, (lv245_1,), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv63_adjoint2 = R.call_tir(cls.add8, (lv63_adjoint1, lv246_1), out_sinfo=R.Tensor((32, 256, 8, 8), dtype="float32"))
            lv247_1 = R.call_tir(cls.conv2d_transpose5, (lv63_adjoint2, conv2d_weight_10), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv62_adjoint1 = R.call_tir(cls.add5, (lv62_adjoint, lv247_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv248_1 = R.call_tir(cls.conv2d16, (lv62_1, lv63_adjoint2), out_sinfo=R.Tensor((256, 128, 4, 4), dtype="float32"))
            conv2d_weight_adjoint9 = R.call_tir(cls.strided_slice3, (lv248_1,), out_sinfo=R.Tensor((256, 128, 3, 3), dtype="float32"))
            lv249_1 = R.call_tir(cls.less2, (lv61_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="bool"))
            lv250_1 = R.call_tir(cls.where2, (lv249_1, R.const(0, "float32"), R.const(1, "float32")), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv61_adjoint = R.call_tir(cls.multiply25, (lv250_1, lv62_adjoint1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv58_adjoint: R.Tensor((32, 128, 16, 16), dtype="float32") = lv61_adjoint
            lv49_adjoint: R.Tensor((32, 128, 16, 16), dtype="float32") = lv61_adjoint
            lv251_1 = R.call_tir(cls.zeros2, R.tuple(), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv252_1 = R.call_tir(cls.zeros2, R.tuple(), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv57_adjoint: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv58_adjoint, lv251_1, lv252_1
            lv173_adjoint: R.Tensor((32, 128, 16, 16), dtype="float32") = lv57_adjoint[0]
            lv176_adjoint: R.Tensor((128,), dtype="float32") = lv57_adjoint[1]
            lv179_adjoint: R.Tensor((128,), dtype="float32") = lv57_adjoint[2]
            lv178_adjoint: R.Tensor((128,), dtype="float32") = lv179_adjoint
            lv165_adjoint = R.call_tir(cls.multiply26, (lv178_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv175_adjoint: R.Tensor((128,), dtype="float32") = lv176_adjoint
            lv162_adjoint = R.call_tir(cls.multiply26, (lv175_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv171_adjoint: R.Tensor((32, 128, 16, 16), dtype="float32") = lv173_adjoint
            lv172_adjoint = R.call_tir(cls.collapse_sum3, (lv173_adjoint,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            bn_beta_adjoint10 = R.call_tir(cls.squeeze2, (lv172_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv169_adjoint = R.call_tir(cls.multiply1, (lv171_adjoint, lv170), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv257_1 = R.call_tir(cls.multiply25, (lv171_adjoint, lv169), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv170_adjoint = R.call_tir(cls.collapse_sum3, (lv257_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            bn_gamma_adjoint10 = R.call_tir(cls.squeeze2, (lv170_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv258_1 = R.call_tir(cls.tir_negative2, (lv169_adjoint,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv259_1 = R.call_tir(cls.multiply25, (lv258_1, lv169), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv164_adjoint = R.call_tir(cls.divide1, (lv169_adjoint, lv168), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv260_1 = R.call_tir(cls.divide1, (lv259_1, lv168), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv168_adjoint = R.call_tir(cls.collapse_sum3, (lv260_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv261_1 = R.call_tir(cls.multiply27, (lv168_adjoint,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv167_adjoint = R.call_tir(cls.divide8, (lv261_1, lv168), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv166_adjoint: R.Tensor((1, 128, 1, 1), dtype="float32") = lv167_adjoint
            lv262_1 = R.call_tir(cls.squeeze2, (lv166_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv165_adjoint1 = R.call_tir(cls.add14, (lv165_adjoint, lv262_1), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv263_1 = R.call_tir(cls.expand_dims1, (lv165_adjoint1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv264_1 = R.call_tir(cls.multiply28, (lv56_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv265_1 = R.call_tir(cls.collapse_sum3, (lv56_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv266_1 = R.call_tir(cls.multiply29, (lv265_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv267_1 = R.call_tir(cls.subtract1, (lv264_1, lv266_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv56_adjoint = R.call_tir(cls.multiply30, (lv263_1, lv267_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv56_adjoint1 = R.call_tir(cls.add5, (lv56_adjoint, lv164_adjoint), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv268_1 = R.call_tir(cls.tir_negative2, (lv164_adjoint,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv163_adjoint = R.call_tir(cls.collapse_sum3, (lv268_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv269_1 = R.call_tir(cls.squeeze2, (lv163_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv162_adjoint1 = R.call_tir(cls.add14, (lv162_adjoint, lv269_1), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv270_1 = R.call_tir(cls.divide9, (lv162_adjoint1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv271_1 = R.call_tir(cls.expand_dims1, (lv270_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv272_1 = R.call_tir(cls.broadcast_to2, (lv271_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv56_adjoint2 = R.call_tir(cls.add5, (lv56_adjoint1, lv272_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv55_adjoint = R.call_tir(cls.conv2d_transpose6, (lv56_adjoint2, conv2d_weight_9), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            conv2d_weight_adjoint10 = R.call_tir(cls.conv2d17, (lv55_1, lv56_adjoint2), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            lv273_1 = R.call_tir(cls.less2, (lv52_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="bool"))
            lv274_1 = R.call_tir(cls.where2, (lv273_1, R.const(0, "float32"), R.const(1, "float32")), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv52_adjoint = R.call_tir(cls.multiply25, (lv274_1, lv55_adjoint), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv275_1 = R.call_tir(cls.zeros2, R.tuple(), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv276_1 = R.call_tir(cls.zeros2, R.tuple(), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv51_adjoint: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv52_adjoint, lv275_1, lv276_1
            lv155_adjoint: R.Tensor((32, 128, 16, 16), dtype="float32") = lv51_adjoint[0]
            lv158_adjoint: R.Tensor((128,), dtype="float32") = lv51_adjoint[1]
            lv161_adjoint: R.Tensor((128,), dtype="float32") = lv51_adjoint[2]
            lv160_adjoint: R.Tensor((128,), dtype="float32") = lv161_adjoint
            lv147_adjoint = R.call_tir(cls.multiply26, (lv160_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv157_adjoint: R.Tensor((128,), dtype="float32") = lv158_adjoint
            lv144_adjoint = R.call_tir(cls.multiply26, (lv157_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv153_adjoint: R.Tensor((32, 128, 16, 16), dtype="float32") = lv155_adjoint
            lv154_adjoint = R.call_tir(cls.collapse_sum3, (lv155_adjoint,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            bn_beta_adjoint11 = R.call_tir(cls.squeeze2, (lv154_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv151_adjoint = R.call_tir(cls.multiply1, (lv153_adjoint, lv152), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv281_1 = R.call_tir(cls.multiply25, (lv153_adjoint, lv151), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv152_adjoint = R.call_tir(cls.collapse_sum3, (lv281_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            bn_gamma_adjoint11 = R.call_tir(cls.squeeze2, (lv152_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv282_1 = R.call_tir(cls.tir_negative2, (lv151_adjoint,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv283_1 = R.call_tir(cls.multiply25, (lv282_1, lv151), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv146_adjoint = R.call_tir(cls.divide1, (lv151_adjoint, lv150), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv284_1 = R.call_tir(cls.divide1, (lv283_1, lv150), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv150_adjoint = R.call_tir(cls.collapse_sum3, (lv284_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv285_1 = R.call_tir(cls.multiply27, (lv150_adjoint,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv149_adjoint = R.call_tir(cls.divide8, (lv285_1, lv150), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv148_adjoint: R.Tensor((1, 128, 1, 1), dtype="float32") = lv149_adjoint
            lv286_1 = R.call_tir(cls.squeeze2, (lv148_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv147_adjoint1 = R.call_tir(cls.add14, (lv147_adjoint, lv286_1), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv287_1 = R.call_tir(cls.expand_dims1, (lv147_adjoint1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv288_1 = R.call_tir(cls.multiply28, (lv50_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv289_1 = R.call_tir(cls.collapse_sum3, (lv50_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv290_1 = R.call_tir(cls.multiply29, (lv289_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv291_1 = R.call_tir(cls.subtract1, (lv288_1, lv290_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv50_adjoint = R.call_tir(cls.multiply30, (lv287_1, lv291_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv50_adjoint1 = R.call_tir(cls.add5, (lv50_adjoint, lv146_adjoint), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv292_1 = R.call_tir(cls.tir_negative2, (lv146_adjoint,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv145_adjoint = R.call_tir(cls.collapse_sum3, (lv292_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv293_1 = R.call_tir(cls.squeeze2, (lv145_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv144_adjoint1 = R.call_tir(cls.add14, (lv144_adjoint, lv293_1), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv294_1 = R.call_tir(cls.divide9, (lv144_adjoint1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv295_1 = R.call_tir(cls.expand_dims1, (lv294_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv296_1 = R.call_tir(cls.broadcast_to2, (lv295_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv50_adjoint2 = R.call_tir(cls.add5, (lv50_adjoint1, lv296_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv297_1 = R.call_tir(cls.conv2d_transpose6, (lv50_adjoint2, conv2d_weight_8), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv49_adjoint1 = R.call_tir(cls.add5, (lv49_adjoint, lv297_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            conv2d_weight_adjoint11 = R.call_tir(cls.conv2d17, (lv49_1, lv50_adjoint2), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            lv298_1 = R.call_tir(cls.less2, (lv48_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="bool"))
            lv299_1 = R.call_tir(cls.where2, (lv298_1, R.const(0, "float32"), R.const(1, "float32")), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv48_adjoint = R.call_tir(cls.multiply25, (lv299_1, lv49_adjoint1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv40_adjoint: R.Tensor((32, 128, 16, 16), dtype="float32") = lv48_adjoint
            lv45_adjoint: R.Tensor((32, 128, 16, 16), dtype="float32") = lv48_adjoint
            lv300_1 = R.call_tir(cls.zeros2, R.tuple(), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv301_1 = R.call_tir(cls.zeros2, R.tuple(), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv44_adjoint: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv45_adjoint, lv300_1, lv301_1
            lv137_adjoint: R.Tensor((32, 128, 16, 16), dtype="float32") = lv44_adjoint[0]
            lv140_adjoint: R.Tensor((128,), dtype="float32") = lv44_adjoint[1]
            lv143_adjoint: R.Tensor((128,), dtype="float32") = lv44_adjoint[2]
            lv142_adjoint: R.Tensor((128,), dtype="float32") = lv143_adjoint
            lv129_adjoint = R.call_tir(cls.multiply26, (lv142_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv139_adjoint: R.Tensor((128,), dtype="float32") = lv140_adjoint
            lv126_adjoint1 = R.call_tir(cls.multiply26, (lv139_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv135_adjoint: R.Tensor((32, 128, 16, 16), dtype="float32") = lv137_adjoint
            lv136_adjoint = R.call_tir(cls.collapse_sum3, (lv137_adjoint,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            bn_beta_adjoint12 = R.call_tir(cls.squeeze2, (lv136_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv133_adjoint = R.call_tir(cls.multiply1, (lv135_adjoint, lv134), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv306_1 = R.call_tir(cls.multiply25, (lv135_adjoint, lv133), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv134_adjoint = R.call_tir(cls.collapse_sum3, (lv306_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            bn_gamma_adjoint12 = R.call_tir(cls.squeeze2, (lv134_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv307_1 = R.call_tir(cls.tir_negative2, (lv133_adjoint,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv308_1 = R.call_tir(cls.multiply25, (lv307_1, lv133), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv128_adjoint1 = R.call_tir(cls.divide1, (lv133_adjoint, lv132), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv309_1 = R.call_tir(cls.divide1, (lv308_1, lv132), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv132_adjoint = R.call_tir(cls.collapse_sum3, (lv309_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv310_1 = R.call_tir(cls.multiply27, (lv132_adjoint,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv131_adjoint = R.call_tir(cls.divide8, (lv310_1, lv132), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv130_adjoint: R.Tensor((1, 128, 1, 1), dtype="float32") = lv131_adjoint
            lv311_1 = R.call_tir(cls.squeeze2, (lv130_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv129_adjoint1 = R.call_tir(cls.add14, (lv129_adjoint, lv311_1), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv312_1 = R.call_tir(cls.expand_dims1, (lv129_adjoint1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv313_1 = R.call_tir(cls.multiply28, (lv43_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv314_1 = R.call_tir(cls.collapse_sum3, (lv43_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv315_1 = R.call_tir(cls.multiply29, (lv314_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv316_1 = R.call_tir(cls.subtract1, (lv313_1, lv315_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv43_adjoint = R.call_tir(cls.multiply30, (lv312_1, lv316_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv43_adjoint1 = R.call_tir(cls.add5, (lv43_adjoint, lv128_adjoint1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv317_1 = R.call_tir(cls.tir_negative2, (lv128_adjoint1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv127_adjoint1 = R.call_tir(cls.collapse_sum3, (lv317_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv318_1 = R.call_tir(cls.squeeze2, (lv127_adjoint1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv126_adjoint2 = R.call_tir(cls.add14, (lv126_adjoint1, lv318_1), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv319_1 = R.call_tir(cls.divide9, (lv126_adjoint2,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv320_1 = R.call_tir(cls.expand_dims1, (lv319_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv321_1 = R.call_tir(cls.broadcast_to2, (lv320_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv43_adjoint2 = R.call_tir(cls.add5, (lv43_adjoint1, lv321_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv31_adjoint = R.call_tir(cls.conv2d_transpose7, (lv43_adjoint2, conv2d_weight_7), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv322_1 = R.call_tir(cls.conv2d18, (lv31_1, lv43_adjoint2), out_sinfo=R.Tensor((128, 64, 2, 2), dtype="float32"))
            conv2d_weight_adjoint12 = R.call_tir(cls.strided_slice4, (lv322_1,), out_sinfo=R.Tensor((128, 64, 1, 1), dtype="float32"))
            lv323_1 = R.call_tir(cls.zeros2, R.tuple(), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv324_1 = R.call_tir(cls.zeros2, R.tuple(), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv39_adjoint: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv40_adjoint, lv323_1, lv324_1
            lv119_adjoint1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv39_adjoint[0]
            lv122_adjoint: R.Tensor((128,), dtype="float32") = lv39_adjoint[1]
            lv125_adjoint1: R.Tensor((128,), dtype="float32") = lv39_adjoint[2]
            lv124_adjoint1: R.Tensor((128,), dtype="float32") = lv125_adjoint1
            lv111_adjoint2 = R.call_tir(cls.multiply26, (lv124_adjoint1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv121_adjoint: R.Tensor((128,), dtype="float32") = lv122_adjoint
            lv108_adjoint = R.call_tir(cls.multiply26, (lv121_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv117_adjoint1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv119_adjoint1
            lv118_adjoint3 = R.call_tir(cls.collapse_sum3, (lv119_adjoint1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            bn_beta_adjoint13 = R.call_tir(cls.squeeze2, (lv118_adjoint3,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv115_adjoint = R.call_tir(cls.multiply1, (lv117_adjoint1, lv116), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv329_1 = R.call_tir(cls.multiply25, (lv117_adjoint1, lv115), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv116_adjoint = R.call_tir(cls.collapse_sum3, (lv329_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            bn_gamma_adjoint13 = R.call_tir(cls.squeeze2, (lv116_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv330_1 = R.call_tir(cls.tir_negative2, (lv115_adjoint,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv331_1 = R.call_tir(cls.multiply25, (lv330_1, lv115), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv110_adjoint1 = R.call_tir(cls.divide1, (lv115_adjoint, lv114), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv332_1 = R.call_tir(cls.divide1, (lv331_1, lv114), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv114_adjoint1 = R.call_tir(cls.collapse_sum3, (lv332_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv333_1 = R.call_tir(cls.multiply27, (lv114_adjoint1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv113_adjoint1 = R.call_tir(cls.divide8, (lv333_1, lv114), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv112_adjoint3: R.Tensor((1, 128, 1, 1), dtype="float32") = lv113_adjoint1
            lv334_1 = R.call_tir(cls.squeeze2, (lv112_adjoint3,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv111_adjoint3 = R.call_tir(cls.add14, (lv111_adjoint2, lv334_1), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv335_1 = R.call_tir(cls.expand_dims1, (lv111_adjoint3,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv336_1 = R.call_tir(cls.multiply28, (lv38_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv337_1 = R.call_tir(cls.collapse_sum3, (lv38_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv338_1 = R.call_tir(cls.multiply29, (lv337_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv339_1 = R.call_tir(cls.subtract1, (lv336_1, lv338_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv38_adjoint = R.call_tir(cls.multiply30, (lv335_1, lv339_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv38_adjoint1 = R.call_tir(cls.add5, (lv38_adjoint, lv110_adjoint1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv340_1 = R.call_tir(cls.tir_negative2, (lv110_adjoint1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv109_adjoint = R.call_tir(cls.collapse_sum3, (lv340_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv341_1 = R.call_tir(cls.squeeze2, (lv109_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv108_adjoint1 = R.call_tir(cls.add14, (lv108_adjoint, lv341_1), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv342_1 = R.call_tir(cls.divide9, (lv108_adjoint1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv343_1 = R.call_tir(cls.expand_dims1, (lv342_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv344_1 = R.call_tir(cls.broadcast_to2, (lv343_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv38_adjoint2 = R.call_tir(cls.add5, (lv38_adjoint1, lv344_1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv37_adjoint = R.call_tir(cls.conv2d_transpose6, (lv38_adjoint2, conv2d_weight_6), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            conv2d_weight_adjoint13 = R.call_tir(cls.conv2d17, (lv37_1, lv38_adjoint2), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            lv345_1 = R.call_tir(cls.less2, (lv34_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="bool"))
            lv346_1 = R.call_tir(cls.where2, (lv345_1, R.const(0, "float32"), R.const(1, "float32")), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv34_adjoint = R.call_tir(cls.multiply25, (lv346_1, lv37_adjoint), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv347_1 = R.call_tir(cls.zeros2, R.tuple(), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv348_1 = R.call_tir(cls.zeros2, R.tuple(), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv33_adjoint: R.Tuple(R.Tensor((32, 128, 16, 16), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = lv34_adjoint, lv347_1, lv348_1
            lv101_adjoint1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv33_adjoint[0]
            lv104_adjoint: R.Tensor((128,), dtype="float32") = lv33_adjoint[1]
            lv107_adjoint1: R.Tensor((128,), dtype="float32") = lv33_adjoint[2]
            lv106_adjoint1: R.Tensor((128,), dtype="float32") = lv107_adjoint1
            lv93_adjoint2 = R.call_tir(cls.multiply26, (lv106_adjoint1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv103_adjoint: R.Tensor((128,), dtype="float32") = lv104_adjoint
            lv90_adjoint = R.call_tir(cls.multiply26, (lv103_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv99_adjoint1: R.Tensor((32, 128, 16, 16), dtype="float32") = lv101_adjoint1
            lv100_adjoint3 = R.call_tir(cls.collapse_sum3, (lv101_adjoint1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            bn_beta_adjoint14 = R.call_tir(cls.squeeze2, (lv100_adjoint3,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv97_adjoint = R.call_tir(cls.multiply1, (lv99_adjoint1, lv98), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv353_1 = R.call_tir(cls.multiply25, (lv99_adjoint1, lv97), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv98_adjoint = R.call_tir(cls.collapse_sum3, (lv353_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            bn_gamma_adjoint14 = R.call_tir(cls.squeeze2, (lv98_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv354_1 = R.call_tir(cls.tir_negative2, (lv97_adjoint,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv355_1 = R.call_tir(cls.multiply25, (lv354_1, lv97), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv92_adjoint1 = R.call_tir(cls.divide1, (lv97_adjoint, lv96), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv356_1 = R.call_tir(cls.divide1, (lv355_1, lv96), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv96_adjoint1 = R.call_tir(cls.collapse_sum3, (lv356_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv357_1 = R.call_tir(cls.multiply27, (lv96_adjoint1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv95_adjoint1 = R.call_tir(cls.divide8, (lv357_1, lv96), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv94_adjoint3: R.Tensor((1, 128, 1, 1), dtype="float32") = lv95_adjoint1
            lv358_1 = R.call_tir(cls.squeeze2, (lv94_adjoint3,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv93_adjoint3 = R.call_tir(cls.add14, (lv93_adjoint2, lv358_1), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv359_1 = R.call_tir(cls.expand_dims1, (lv93_adjoint3,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv360 = R.call_tir(cls.multiply28, (lv32_1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv361 = R.call_tir(cls.collapse_sum3, (lv32_1,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv362 = R.call_tir(cls.multiply29, (lv361,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv363 = R.call_tir(cls.subtract1, (lv360, lv362), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv32_adjoint = R.call_tir(cls.multiply30, (lv359_1, lv363), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv32_adjoint1 = R.call_tir(cls.add5, (lv32_adjoint, lv92_adjoint1), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv364 = R.call_tir(cls.tir_negative2, (lv92_adjoint1,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv91_adjoint = R.call_tir(cls.collapse_sum3, (lv364,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv365 = R.call_tir(cls.squeeze2, (lv91_adjoint,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv90_adjoint1 = R.call_tir(cls.add14, (lv90_adjoint, lv365), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv366 = R.call_tir(cls.divide9, (lv90_adjoint1,), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv367 = R.call_tir(cls.expand_dims1, (lv366,), out_sinfo=R.Tensor((1, 128, 1, 1), dtype="float32"))
            lv368 = R.call_tir(cls.broadcast_to2, (lv367,), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv32_adjoint2 = R.call_tir(cls.add5, (lv32_adjoint1, lv368), out_sinfo=R.Tensor((32, 128, 16, 16), dtype="float32"))
            lv369 = R.call_tir(cls.conv2d_transpose8, (lv32_adjoint2, conv2d_weight_5), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv31_adjoint1 = R.call_tir(cls.add2, (lv31_adjoint, lv369), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv370 = R.call_tir(cls.conv2d19, (lv31_1, lv32_adjoint2), out_sinfo=R.Tensor((128, 64, 4, 4), dtype="float32"))
            conv2d_weight_adjoint14 = R.call_tir(cls.strided_slice5, (lv370,), out_sinfo=R.Tensor((128, 64, 3, 3), dtype="float32"))
            lv371 = R.call_tir(cls.less3, (lv30_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="bool"))
            lv372 = R.call_tir(cls.where3, (lv371, R.const(0, "float32"), R.const(1, "float32")), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv30_adjoint = R.call_tir(cls.multiply31, (lv372, lv31_adjoint1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv27_adjoint: R.Tensor((32, 64, 32, 32), dtype="float32") = lv30_adjoint
            lv18_adjoint: R.Tensor((32, 64, 32, 32), dtype="float32") = lv30_adjoint
            lv373 = R.call_tir(cls.zeros3, R.tuple(), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv374 = R.call_tir(cls.zeros3, R.tuple(), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv26_adjoint: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv27_adjoint, lv373, lv374
            lv83_adjoint1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv26_adjoint[0]
            lv86_adjoint1: R.Tensor((64,), dtype="float32") = lv26_adjoint[1]
            lv89_adjoint1: R.Tensor((64,), dtype="float32") = lv26_adjoint[2]
            lv88_adjoint1: R.Tensor((64,), dtype="float32") = lv89_adjoint1
            lv75_adjoint1 = R.call_tir(cls.multiply32, (lv88_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv85_adjoint: R.Tensor((64,), dtype="float32") = lv86_adjoint1
            lv72_adjoint = R.call_tir(cls.multiply32, (lv85_adjoint,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv81_adjoint3: R.Tensor((32, 64, 32, 32), dtype="float32") = lv83_adjoint1
            lv82_adjoint1 = R.call_tir(cls.collapse_sum4, (lv83_adjoint1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            bn_beta_adjoint15 = R.call_tir(cls.squeeze3, (lv82_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv79_adjoint1 = R.call_tir(cls.multiply, (lv81_adjoint3, lv80), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv379 = R.call_tir(cls.multiply31, (lv81_adjoint3, lv79), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv80_adjoint2 = R.call_tir(cls.collapse_sum4, (lv379,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            bn_gamma_adjoint15 = R.call_tir(cls.squeeze3, (lv80_adjoint2,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv380 = R.call_tir(cls.tir_negative3, (lv79_adjoint1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv381 = R.call_tir(cls.multiply31, (lv380, lv79), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv74_adjoint3 = R.call_tir(cls.divide, (lv79_adjoint1, lv78), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv382 = R.call_tir(cls.divide, (lv381, lv78), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv78_adjoint = R.call_tir(cls.collapse_sum4, (lv382,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv383 = R.call_tir(cls.multiply33, (lv78_adjoint,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv77_adjoint = R.call_tir(cls.divide10, (lv383, lv78), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv76_adjoint1: R.Tensor((1, 64, 1, 1), dtype="float32") = lv77_adjoint
            lv384 = R.call_tir(cls.squeeze3, (lv76_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv75_adjoint2 = R.call_tir(cls.add13, (lv75_adjoint1, lv384), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv385 = R.call_tir(cls.expand_dims, (lv75_adjoint2,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv386 = R.call_tir(cls.multiply34, (lv25_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv387 = R.call_tir(cls.collapse_sum4, (lv25_1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv388 = R.call_tir(cls.multiply35, (lv387,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv389 = R.call_tir(cls.subtract, (lv386, lv388), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv25_adjoint = R.call_tir(cls.multiply36, (lv385, lv389), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv25_adjoint1 = R.call_tir(cls.add2, (lv25_adjoint, lv74_adjoint3), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv390 = R.call_tir(cls.tir_negative3, (lv74_adjoint3,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv73_adjoint = R.call_tir(cls.collapse_sum4, (lv390,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv391 = R.call_tir(cls.squeeze3, (lv73_adjoint,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv72_adjoint1 = R.call_tir(cls.add13, (lv72_adjoint, lv391), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv392 = R.call_tir(cls.divide11, (lv72_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv393 = R.call_tir(cls.expand_dims, (lv392,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv394 = R.call_tir(cls.broadcast_to3, (lv393,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv25_adjoint2 = R.call_tir(cls.add2, (lv25_adjoint1, lv394), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv24_adjoint = R.call_tir(cls.conv2d_transpose9, (lv25_adjoint2, conv2d_weight_4), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            conv2d_weight_adjoint15 = R.call_tir(cls.conv2d20, (lv24_1, lv25_adjoint2), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            lv395 = R.call_tir(cls.less3, (lv21_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="bool"))
            lv396 = R.call_tir(cls.where3, (lv395, R.const(0, "float32"), R.const(1, "float32")), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv21_adjoint = R.call_tir(cls.multiply31, (lv396, lv24_adjoint), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv397 = R.call_tir(cls.zeros3, R.tuple(), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv398 = R.call_tir(cls.zeros3, R.tuple(), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv20_adjoint: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv21_adjoint, lv397, lv398
            lv65_adjoint1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv20_adjoint[0]
            lv68_adjoint1: R.Tensor((64,), dtype="float32") = lv20_adjoint[1]
            lv71_adjoint1: R.Tensor((64,), dtype="float32") = lv20_adjoint[2]
            lv70_adjoint1: R.Tensor((64,), dtype="float32") = lv71_adjoint1
            lv57_adjoint1 = R.call_tir(cls.multiply32, (lv70_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv67_adjoint: R.Tensor((64,), dtype="float32") = lv68_adjoint1
            lv54_adjoint = R.call_tir(cls.multiply32, (lv67_adjoint,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv63_adjoint3: R.Tensor((32, 64, 32, 32), dtype="float32") = lv65_adjoint1
            lv64_adjoint1 = R.call_tir(cls.collapse_sum4, (lv65_adjoint1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            bn_beta_adjoint16 = R.call_tir(cls.squeeze3, (lv64_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv61_adjoint1 = R.call_tir(cls.multiply, (lv63_adjoint3, lv62), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv403 = R.call_tir(cls.multiply31, (lv63_adjoint3, lv61), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv62_adjoint2 = R.call_tir(cls.collapse_sum4, (lv403,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            bn_gamma_adjoint16 = R.call_tir(cls.squeeze3, (lv62_adjoint2,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv404 = R.call_tir(cls.tir_negative3, (lv61_adjoint1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv405 = R.call_tir(cls.multiply31, (lv404, lv61), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv56_adjoint3 = R.call_tir(cls.divide, (lv61_adjoint1, lv60), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv406 = R.call_tir(cls.divide, (lv405, lv60), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv60_adjoint = R.call_tir(cls.collapse_sum4, (lv406,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv407 = R.call_tir(cls.multiply33, (lv60_adjoint,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv59_adjoint = R.call_tir(cls.divide10, (lv407, lv60), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv58_adjoint1: R.Tensor((1, 64, 1, 1), dtype="float32") = lv59_adjoint
            lv408 = R.call_tir(cls.squeeze3, (lv58_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv57_adjoint2 = R.call_tir(cls.add13, (lv57_adjoint1, lv408), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv409 = R.call_tir(cls.expand_dims, (lv57_adjoint2,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv410 = R.call_tir(cls.multiply34, (lv19_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv411 = R.call_tir(cls.collapse_sum4, (lv19_1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv412 = R.call_tir(cls.multiply35, (lv411,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv413 = R.call_tir(cls.subtract, (lv410, lv412), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv19_adjoint = R.call_tir(cls.multiply36, (lv409, lv413), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv19_adjoint1 = R.call_tir(cls.add2, (lv19_adjoint, lv56_adjoint3), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv414 = R.call_tir(cls.tir_negative3, (lv56_adjoint3,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv55_adjoint1 = R.call_tir(cls.collapse_sum4, (lv414,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv415 = R.call_tir(cls.squeeze3, (lv55_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv54_adjoint1 = R.call_tir(cls.add13, (lv54_adjoint, lv415), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv416 = R.call_tir(cls.divide11, (lv54_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv417 = R.call_tir(cls.expand_dims, (lv416,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv418 = R.call_tir(cls.broadcast_to3, (lv417,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv19_adjoint2 = R.call_tir(cls.add2, (lv19_adjoint1, lv418), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv419 = R.call_tir(cls.conv2d_transpose9, (lv19_adjoint2, conv2d_weight_3), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv18_adjoint1 = R.call_tir(cls.add2, (lv18_adjoint, lv419), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            conv2d_weight_adjoint16 = R.call_tir(cls.conv2d20, (lv18_1, lv19_adjoint2), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            lv420 = R.call_tir(cls.less3, (lv17_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="bool"))
            lv421 = R.call_tir(cls.where3, (lv420, R.const(0, "float32"), R.const(1, "float32")), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv17_adjoint = R.call_tir(cls.multiply31, (lv421, lv18_adjoint1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv14_adjoint: R.Tensor((32, 64, 32, 32), dtype="float32") = lv17_adjoint
            lv5_adjoint: R.Tensor((32, 64, 32, 32), dtype="float32") = lv17_adjoint
            lv422 = R.call_tir(cls.zeros3, R.tuple(), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv423 = R.call_tir(cls.zeros3, R.tuple(), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv13_adjoint: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv14_adjoint, lv422, lv423
            lv47_adjoint: R.Tensor((32, 64, 32, 32), dtype="float32") = lv13_adjoint[0]
            lv50_adjoint3: R.Tensor((64,), dtype="float32") = lv13_adjoint[1]
            lv53_adjoint: R.Tensor((64,), dtype="float32") = lv13_adjoint[2]
            lv52_adjoint1: R.Tensor((64,), dtype="float32") = lv53_adjoint
            lv39_adjoint1 = R.call_tir(cls.multiply32, (lv52_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv49_adjoint2: R.Tensor((64,), dtype="float32") = lv50_adjoint3
            lv36_adjoint = R.call_tir(cls.multiply32, (lv49_adjoint2,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv45_adjoint1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv47_adjoint
            lv46_adjoint = R.call_tir(cls.collapse_sum4, (lv47_adjoint,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            bn_beta_adjoint17 = R.call_tir(cls.squeeze3, (lv46_adjoint,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv43_adjoint3 = R.call_tir(cls.multiply, (lv45_adjoint1, lv44), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv428 = R.call_tir(cls.multiply31, (lv45_adjoint1, lv43), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv44_adjoint1 = R.call_tir(cls.collapse_sum4, (lv428,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            bn_gamma_adjoint17 = R.call_tir(cls.squeeze3, (lv44_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv429 = R.call_tir(cls.tir_negative3, (lv43_adjoint3,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv430 = R.call_tir(cls.multiply31, (lv429, lv43), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv38_adjoint3 = R.call_tir(cls.divide, (lv43_adjoint3, lv42), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv431 = R.call_tir(cls.divide, (lv430, lv42), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv42_adjoint = R.call_tir(cls.collapse_sum4, (lv431,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv432 = R.call_tir(cls.multiply33, (lv42_adjoint,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv41_adjoint = R.call_tir(cls.divide10, (lv432, lv42), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv40_adjoint1: R.Tensor((1, 64, 1, 1), dtype="float32") = lv41_adjoint
            lv433 = R.call_tir(cls.squeeze3, (lv40_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv39_adjoint2 = R.call_tir(cls.add13, (lv39_adjoint1, lv433), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv434 = R.call_tir(cls.expand_dims, (lv39_adjoint2,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv435 = R.call_tir(cls.multiply34, (lv12_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv436 = R.call_tir(cls.collapse_sum4, (lv12_1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv437 = R.call_tir(cls.multiply35, (lv436,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv438 = R.call_tir(cls.subtract, (lv435, lv437), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv12_adjoint = R.call_tir(cls.multiply36, (lv434, lv438), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv12_adjoint1 = R.call_tir(cls.add2, (lv12_adjoint, lv38_adjoint3), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv439 = R.call_tir(cls.tir_negative3, (lv38_adjoint3,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv37_adjoint1 = R.call_tir(cls.collapse_sum4, (lv439,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv440 = R.call_tir(cls.squeeze3, (lv37_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv36_adjoint1 = R.call_tir(cls.add13, (lv36_adjoint, lv440), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv441 = R.call_tir(cls.divide11, (lv36_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv442 = R.call_tir(cls.expand_dims, (lv441,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv443 = R.call_tir(cls.broadcast_to3, (lv442,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv12_adjoint2 = R.call_tir(cls.add2, (lv12_adjoint1, lv443), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv11_adjoint = R.call_tir(cls.conv2d_transpose9, (lv12_adjoint2, conv2d_weight_2), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            conv2d_weight_adjoint17 = R.call_tir(cls.conv2d20, (lv11_1, lv12_adjoint2), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            lv444 = R.call_tir(cls.less3, (lv8_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="bool"))
            lv445 = R.call_tir(cls.where3, (lv444, R.const(0, "float32"), R.const(1, "float32")), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv8_adjoint = R.call_tir(cls.multiply31, (lv445, lv11_adjoint), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv446 = R.call_tir(cls.zeros3, R.tuple(), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv447 = R.call_tir(cls.zeros3, R.tuple(), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv7_adjoint: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv8_adjoint, lv446, lv447
            lv29_adjoint: R.Tensor((32, 64, 32, 32), dtype="float32") = lv7_adjoint[0]
            lv32_adjoint3: R.Tensor((64,), dtype="float32") = lv7_adjoint[1]
            lv35_adjoint: R.Tensor((64,), dtype="float32") = lv7_adjoint[2]
            lv34_adjoint1: R.Tensor((64,), dtype="float32") = lv35_adjoint
            lv21_adjoint1 = R.call_tir(cls.multiply32, (lv34_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv31_adjoint2: R.Tensor((64,), dtype="float32") = lv32_adjoint3
            lv18_adjoint2 = R.call_tir(cls.multiply32, (lv31_adjoint2,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv27_adjoint1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv29_adjoint
            lv28_adjoint = R.call_tir(cls.collapse_sum4, (lv29_adjoint,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            bn_beta_adjoint18 = R.call_tir(cls.squeeze3, (lv28_adjoint,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv25_adjoint3 = R.call_tir(cls.multiply, (lv27_adjoint1, lv26), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv452 = R.call_tir(cls.multiply31, (lv27_adjoint1, lv25), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv26_adjoint1 = R.call_tir(cls.collapse_sum4, (lv452,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            bn_gamma_adjoint18 = R.call_tir(cls.squeeze3, (lv26_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv453 = R.call_tir(cls.tir_negative3, (lv25_adjoint3,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv454 = R.call_tir(cls.multiply31, (lv453, lv25), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv20_adjoint1 = R.call_tir(cls.divide, (lv25_adjoint3, lv24), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv455 = R.call_tir(cls.divide, (lv454, lv24), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv24_adjoint1 = R.call_tir(cls.collapse_sum4, (lv455,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv456 = R.call_tir(cls.multiply33, (lv24_adjoint1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv23_adjoint = R.call_tir(cls.divide10, (lv456, lv24), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv22_adjoint: R.Tensor((1, 64, 1, 1), dtype="float32") = lv23_adjoint
            lv457 = R.call_tir(cls.squeeze3, (lv22_adjoint,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv21_adjoint2 = R.call_tir(cls.add13, (lv21_adjoint1, lv457), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv458 = R.call_tir(cls.expand_dims, (lv21_adjoint2,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv459 = R.call_tir(cls.multiply34, (lv6_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv460 = R.call_tir(cls.collapse_sum4, (lv6_1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv461 = R.call_tir(cls.multiply35, (lv460,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv462 = R.call_tir(cls.subtract, (lv459, lv461), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv6_adjoint = R.call_tir(cls.multiply36, (lv458, lv462), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv6_adjoint1 = R.call_tir(cls.add2, (lv6_adjoint, lv20_adjoint1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv463 = R.call_tir(cls.tir_negative3, (lv20_adjoint1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv19_adjoint3 = R.call_tir(cls.collapse_sum4, (lv463,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv464 = R.call_tir(cls.squeeze3, (lv19_adjoint3,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv18_adjoint3 = R.call_tir(cls.add13, (lv18_adjoint2, lv464), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv465 = R.call_tir(cls.divide11, (lv18_adjoint3,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv466 = R.call_tir(cls.expand_dims, (lv465,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv467 = R.call_tir(cls.broadcast_to3, (lv466,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv6_adjoint2 = R.call_tir(cls.add2, (lv6_adjoint1, lv467), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv468 = R.call_tir(cls.conv2d_transpose9, (lv6_adjoint2, conv2d_weight_1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv5_adjoint1 = R.call_tir(cls.add2, (lv5_adjoint, lv468), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            conv2d_weight_adjoint18 = R.call_tir(cls.conv2d20, (lv5_1, lv6_adjoint2), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            lv469 = R.call_tir(cls.less3, (lv2_1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="bool"))
            lv470 = R.call_tir(cls.where3, (lv469, R.const(0, "float32"), R.const(1, "float32")), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv2_adjoint = R.call_tir(cls.multiply31, (lv470, lv5_adjoint1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv471 = R.call_tir(cls.zeros3, R.tuple(), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv472 = R.call_tir(cls.zeros3, R.tuple(), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv1_adjoint: R.Tuple(R.Tensor((32, 64, 32, 32), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = lv2_adjoint, lv471, lv472
            lv11_adjoint1: R.Tensor((32, 64, 32, 32), dtype="float32") = lv1_adjoint[0]
            lv14_adjoint1: R.Tensor((64,), dtype="float32") = lv1_adjoint[1]
            lv17_adjoint1: R.Tensor((64,), dtype="float32") = lv1_adjoint[2]
            lv16_adjoint: R.Tensor((64,), dtype="float32") = lv17_adjoint1
            lv3_adjoint = R.call_tir(cls.multiply32, (lv16_adjoint,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv13_adjoint1: R.Tensor((64,), dtype="float32") = lv14_adjoint1
            lv_adjoint1 = R.call_tir(cls.multiply32, (lv13_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv9_adjoint: R.Tensor((32, 64, 32, 32), dtype="float32") = lv11_adjoint1
            lv10_adjoint = R.call_tir(cls.collapse_sum4, (lv11_adjoint1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            bn_beta_adjoint19 = R.call_tir(cls.squeeze3, (lv10_adjoint,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv7_adjoint1 = R.call_tir(cls.multiply, (lv9_adjoint, lv8), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv477 = R.call_tir(cls.multiply31, (lv9_adjoint, lv7), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv8_adjoint1 = R.call_tir(cls.collapse_sum4, (lv477,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            bn_gamma_adjoint19 = R.call_tir(cls.squeeze3, (lv8_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv478 = R.call_tir(cls.tir_negative3, (lv7_adjoint1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv479 = R.call_tir(cls.multiply31, (lv478, lv7), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv2_adjoint1 = R.call_tir(cls.divide, (lv7_adjoint1, lv6), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv480 = R.call_tir(cls.divide, (lv479, lv6), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv6_adjoint3 = R.call_tir(cls.collapse_sum4, (lv480,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv481 = R.call_tir(cls.multiply33, (lv6_adjoint3,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv5_adjoint2 = R.call_tir(cls.divide10, (lv481, lv6), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv4_adjoint: R.Tensor((1, 64, 1, 1), dtype="float32") = lv5_adjoint2
            lv482 = R.call_tir(cls.squeeze3, (lv4_adjoint,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv3_adjoint1 = R.call_tir(cls.add13, (lv3_adjoint, lv482), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv483 = R.call_tir(cls.expand_dims, (lv3_adjoint1,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv484 = R.call_tir(cls.multiply34, (lv,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv485 = R.call_tir(cls.collapse_sum4, (lv,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv486 = R.call_tir(cls.multiply35, (lv485,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv487 = R.call_tir(cls.subtract, (lv484, lv486), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv_adjoint2 = R.call_tir(cls.multiply36, (lv483, lv487), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv_adjoint3 = R.call_tir(cls.add2, (lv_adjoint2, lv2_adjoint1), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv488 = R.call_tir(cls.tir_negative3, (lv2_adjoint1,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv1_adjoint1 = R.call_tir(cls.collapse_sum4, (lv488,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv489 = R.call_tir(cls.squeeze3, (lv1_adjoint1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv_adjoint4 = R.call_tir(cls.add13, (lv_adjoint1, lv489), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv490 = R.call_tir(cls.divide11, (lv_adjoint4,), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv491 = R.call_tir(cls.expand_dims, (lv490,), out_sinfo=R.Tensor((1, 64, 1, 1), dtype="float32"))
            lv492 = R.call_tir(cls.broadcast_to3, (lv491,), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            lv_adjoint5 = R.call_tir(cls.add2, (lv_adjoint3, lv492), out_sinfo=R.Tensor((32, 64, 32, 32), dtype="float32"))
            conv2d_weight_adjoint19 = R.call_tir(cls.conv2d21, (input, lv_adjoint5), out_sinfo=R.Tensor((64, 3, 3, 3), dtype="float32"))
            conv2d_weight_adjoint_out: R.Tensor((64, 3, 3, 3), dtype="float32") = conv2d_weight_adjoint19
            bn_gamma_adjoint_out: R.Tensor((64,), dtype="float32") = bn_gamma_adjoint19
            bn_beta_adjoint_out: R.Tensor((64,), dtype="float32") = bn_beta_adjoint19
            conv2d_weight_adjoint_out1: R.Tensor((64, 64, 3, 3), dtype="float32") = conv2d_weight_adjoint18
            bn_gamma_adjoint_out1: R.Tensor((64,), dtype="float32") = bn_gamma_adjoint18
            bn_beta_adjoint_out1: R.Tensor((64,), dtype="float32") = bn_beta_adjoint18
            conv2d_weight_adjoint_out2: R.Tensor((64, 64, 3, 3), dtype="float32") = conv2d_weight_adjoint17
            bn_gamma_adjoint_out2: R.Tensor((64,), dtype="float32") = bn_gamma_adjoint17
            bn_beta_adjoint_out2: R.Tensor((64,), dtype="float32") = bn_beta_adjoint17
            conv2d_weight_adjoint_out3: R.Tensor((64, 64, 3, 3), dtype="float32") = conv2d_weight_adjoint16
            bn_gamma_adjoint_out3: R.Tensor((64,), dtype="float32") = bn_gamma_adjoint16
            bn_beta_adjoint_out3: R.Tensor((64,), dtype="float32") = bn_beta_adjoint16
            conv2d_weight_adjoint_out4: R.Tensor((64, 64, 3, 3), dtype="float32") = conv2d_weight_adjoint15
            bn_gamma_adjoint_out4: R.Tensor((64,), dtype="float32") = bn_gamma_adjoint15
            bn_beta_adjoint_out4: R.Tensor((64,), dtype="float32") = bn_beta_adjoint15
            conv2d_weight_adjoint_out5: R.Tensor((128, 64, 3, 3), dtype="float32") = conv2d_weight_adjoint14
            bn_gamma_adjoint_out5: R.Tensor((128,), dtype="float32") = bn_gamma_adjoint14
            bn_beta_adjoint_out5: R.Tensor((128,), dtype="float32") = bn_beta_adjoint14
            conv2d_weight_adjoint_out6: R.Tensor((128, 128, 3, 3), dtype="float32") = conv2d_weight_adjoint13
            bn_gamma_adjoint_out6: R.Tensor((128,), dtype="float32") = bn_gamma_adjoint13
            bn_beta_adjoint_out6: R.Tensor((128,), dtype="float32") = bn_beta_adjoint13
            conv2d_weight_adjoint_out7: R.Tensor((128, 64, 1, 1), dtype="float32") = conv2d_weight_adjoint12
            bn_gamma_adjoint_out7: R.Tensor((128,), dtype="float32") = bn_gamma_adjoint12
            bn_beta_adjoint_out7: R.Tensor((128,), dtype="float32") = bn_beta_adjoint12
            conv2d_weight_adjoint_out8: R.Tensor((128, 128, 3, 3), dtype="float32") = conv2d_weight_adjoint11
            bn_gamma_adjoint_out8: R.Tensor((128,), dtype="float32") = bn_gamma_adjoint11
            bn_beta_adjoint_out8: R.Tensor((128,), dtype="float32") = bn_beta_adjoint11
            conv2d_weight_adjoint_out9: R.Tensor((128, 128, 3, 3), dtype="float32") = conv2d_weight_adjoint10
            bn_gamma_adjoint_out9: R.Tensor((128,), dtype="float32") = bn_gamma_adjoint10
            bn_beta_adjoint_out9: R.Tensor((128,), dtype="float32") = bn_beta_adjoint10
            conv2d_weight_adjoint_out10: R.Tensor((256, 128, 3, 3), dtype="float32") = conv2d_weight_adjoint9
            bn_gamma_adjoint_out10: R.Tensor((256,), dtype="float32") = bn_gamma_adjoint9
            bn_beta_adjoint_out10: R.Tensor((256,), dtype="float32") = bn_beta_adjoint9
            conv2d_weight_adjoint_out11: R.Tensor((256, 256, 3, 3), dtype="float32") = conv2d_weight_adjoint8
            bn_gamma_adjoint_out11: R.Tensor((256,), dtype="float32") = bn_gamma_adjoint8
            bn_beta_adjoint_out11: R.Tensor((256,), dtype="float32") = bn_beta_adjoint8
            conv2d_weight_adjoint_out12: R.Tensor((256, 128, 1, 1), dtype="float32") = conv2d_weight_adjoint7
            bn_gamma_adjoint_out12: R.Tensor((256,), dtype="float32") = bn_gamma_adjoint7
            bn_beta_adjoint_out12: R.Tensor((256,), dtype="float32") = bn_beta_adjoint7
            conv2d_weight_adjoint_out13: R.Tensor((256, 256, 3, 3), dtype="float32") = conv2d_weight_adjoint6
            bn_gamma_adjoint_out13: R.Tensor((256,), dtype="float32") = bn_gamma_adjoint6
            bn_beta_adjoint_out13: R.Tensor((256,), dtype="float32") = bn_beta_adjoint6
            conv2d_weight_adjoint_out14: R.Tensor((256, 256, 3, 3), dtype="float32") = conv2d_weight_adjoint5
            bn_gamma_adjoint_out14: R.Tensor((256,), dtype="float32") = bn_gamma_adjoint5
            bn_beta_adjoint_out14: R.Tensor((256,), dtype="float32") = bn_beta_adjoint5
            conv2d_weight_adjoint_out15: R.Tensor((512, 256, 3, 3), dtype="float32") = conv2d_weight_adjoint4
            bn_gamma_adjoint_out15: R.Tensor((512,), dtype="float32") = bn_gamma_adjoint4
            bn_beta_adjoint_out15: R.Tensor((512,), dtype="float32") = bn_beta_adjoint4
            conv2d_weight_adjoint_out16: R.Tensor((512, 512, 3, 3), dtype="float32") = conv2d_weight_adjoint3
            bn_gamma_adjoint_out16: R.Tensor((512,), dtype="float32") = bn_gamma_adjoint3
            bn_beta_adjoint_out16: R.Tensor((512,), dtype="float32") = bn_beta_adjoint3
            conv2d_weight_adjoint_out17: R.Tensor((512, 256, 1, 1), dtype="float32") = conv2d_weight_adjoint2
            bn_gamma_adjoint_out17: R.Tensor((512,), dtype="float32") = bn_gamma_adjoint2
            bn_beta_adjoint_out17: R.Tensor((512,), dtype="float32") = bn_beta_adjoint2
            conv2d_weight_adjoint_out18: R.Tensor((512, 512, 3, 3), dtype="float32") = conv2d_weight_adjoint1
            bn_gamma_adjoint_out18: R.Tensor((512,), dtype="float32") = bn_gamma_adjoint1
            bn_beta_adjoint_out18: R.Tensor((512,), dtype="float32") = bn_beta_adjoint1
            conv2d_weight_adjoint_out19: R.Tensor((512, 512, 3, 3), dtype="float32") = conv2d_weight_adjoint
            bn_gamma_adjoint_out19: R.Tensor((512,), dtype="float32") = bn_gamma_adjoint
            bn_beta_adjoint_out19: R.Tensor((512,), dtype="float32") = bn_beta_adjoint
            ln_weight_adjoint_out: R.Tensor((512, 10), dtype="float32") = ln_weight_adjoint
            ln_bias_adjoint_out: R.Tensor((10,), dtype="float32") = ln_bias_adjoint
            R.output(gv1, gv2, gv3, gv4, gv5, gv6, gv7, gv8, gv9, gv10, gv11, gv12, gv13, gv14, gv15, gv16, gv17, gv18, gv19, gv20, gv21, gv22, gv23, gv24, gv25, gv26, gv27, gv28, gv29, gv30, gv31, gv32, gv33, gv34, gv35, gv36, gv37, gv38, gv39, gv40, gv_1, conv2d_weight_adjoint_out, bn_gamma_adjoint_out, bn_beta_adjoint_out, conv2d_weight_adjoint_out1, bn_gamma_adjoint_out1, bn_beta_adjoint_out1, conv2d_weight_adjoint_out2, bn_gamma_adjoint_out2, bn_beta_adjoint_out2, conv2d_weight_adjoint_out3, bn_gamma_adjoint_out3, bn_beta_adjoint_out3, conv2d_weight_adjoint_out4, bn_gamma_adjoint_out4, bn_beta_adjoint_out4, conv2d_weight_adjoint_out5, bn_gamma_adjoint_out5, bn_beta_adjoint_out5, conv2d_weight_adjoint_out6, bn_gamma_adjoint_out6, bn_beta_adjoint_out6, conv2d_weight_adjoint_out7, bn_gamma_adjoint_out7, bn_beta_adjoint_out7, conv2d_weight_adjoint_out8, bn_gamma_adjoint_out8, bn_beta_adjoint_out8, conv2d_weight_adjoint_out9, bn_gamma_adjoint_out9, bn_beta_adjoint_out9, conv2d_weight_adjoint_out10, bn_gamma_adjoint_out10, bn_beta_adjoint_out10, conv2d_weight_adjoint_out11, bn_gamma_adjoint_out11, bn_beta_adjoint_out11, conv2d_weight_adjoint_out12, bn_gamma_adjoint_out12, bn_beta_adjoint_out12, conv2d_weight_adjoint_out13, bn_gamma_adjoint_out13, bn_beta_adjoint_out13, conv2d_weight_adjoint_out14, bn_gamma_adjoint_out14, bn_beta_adjoint_out14, conv2d_weight_adjoint_out15, bn_gamma_adjoint_out15, bn_beta_adjoint_out15, conv2d_weight_adjoint_out16, bn_gamma_adjoint_out16, bn_beta_adjoint_out16, conv2d_weight_adjoint_out17, bn_gamma_adjoint_out17, bn_beta_adjoint_out17, conv2d_weight_adjoint_out18, bn_gamma_adjoint_out18, bn_beta_adjoint_out18, conv2d_weight_adjoint_out19, bn_gamma_adjoint_out19, bn_beta_adjoint_out19, ln_weight_adjoint_out, ln_bias_adjoint_out)
        return ((gv_1, gv1, gv2, gv3, gv4, gv5, gv6, gv7, gv8, gv9, gv10, gv11, gv12, gv13, gv14, gv15, gv16, gv17, gv18, gv19, gv20, gv21, gv22, gv23, gv24, gv25, gv26, gv27, gv28, gv29, gv30, gv31, gv32, gv33, gv34, gv35, gv36, gv37, gv38, gv39, gv40), (conv2d_weight_adjoint_out, bn_gamma_adjoint_out, bn_beta_adjoint_out, conv2d_weight_adjoint_out1, bn_gamma_adjoint_out1, bn_beta_adjoint_out1, conv2d_weight_adjoint_out2, bn_gamma_adjoint_out2, bn_beta_adjoint_out2, conv2d_weight_adjoint_out3, bn_gamma_adjoint_out3, bn_beta_adjoint_out3, conv2d_weight_adjoint_out4, bn_gamma_adjoint_out4, bn_beta_adjoint_out4, conv2d_weight_adjoint_out5, bn_gamma_adjoint_out5, bn_beta_adjoint_out5, conv2d_weight_adjoint_out6, bn_gamma_adjoint_out6, bn_beta_adjoint_out6, conv2d_weight_adjoint_out7, bn_gamma_adjoint_out7, bn_beta_adjoint_out7, conv2d_weight_adjoint_out8, bn_gamma_adjoint_out8, bn_beta_adjoint_out8, conv2d_weight_adjoint_out9, bn_gamma_adjoint_out9, bn_beta_adjoint_out9, conv2d_weight_adjoint_out10, bn_gamma_adjoint_out10, bn_beta_adjoint_out10, conv2d_weight_adjoint_out11, bn_gamma_adjoint_out11, bn_beta_adjoint_out11, conv2d_weight_adjoint_out12, bn_gamma_adjoint_out12, bn_beta_adjoint_out12, conv2d_weight_adjoint_out13, bn_gamma_adjoint_out13, bn_beta_adjoint_out13, conv2d_weight_adjoint_out14, bn_gamma_adjoint_out14, bn_beta_adjoint_out14, conv2d_weight_adjoint_out15, bn_gamma_adjoint_out15, bn_beta_adjoint_out15, conv2d_weight_adjoint_out16, bn_gamma_adjoint_out16, bn_beta_adjoint_out16, conv2d_weight_adjoint_out17, bn_gamma_adjoint_out17, bn_beta_adjoint_out17, conv2d_weight_adjoint_out18, bn_gamma_adjoint_out18, bn_beta_adjoint_out18, conv2d_weight_adjoint_out19, bn_gamma_adjoint_out19, bn_beta_adjoint_out19, ln_weight_adjoint_out, ln_bias_adjoint_out))

    @R.function
    def optimizer(params: R.Tuple(R.Tensor((64, 3, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((128, 64, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 64, 1, 1), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((256, 128, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 128, 1, 1), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((512, 256, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 256, 1, 1), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 10), dtype="float32"), R.Tensor((10,), dtype="float32")), gradients: R.Tuple(R.Tensor((64, 3, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((128, 64, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 64, 1, 1), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((256, 128, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 128, 1, 1), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((512, 256, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 256, 1, 1), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 10), dtype="float32"), R.Tensor((10,), dtype="float32")), optim_states: R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((64, 3, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((128, 64, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 64, 1, 1), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((256, 128, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 128, 1, 1), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((512, 256, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 256, 1, 1), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 10), dtype="float32"), R.Tensor((10,), dtype="float32"))) -> R.Tuple(R.Tuple(R.Tensor((64, 3, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((128, 64, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 64, 1, 1), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((256, 128, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 128, 1, 1), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((512, 256, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 256, 1, 1), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 10), dtype="float32"), R.Tensor((10,), dtype="float32")), R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((64, 3, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((128, 64, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 64, 1, 1), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((256, 128, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 128, 1, 1), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((512, 256, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 256, 1, 1), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 10), dtype="float32"), R.Tensor((10,), dtype="float32"))):
        cls = Module
        with R.dataflow():
            num_steps: R.Tensor((), dtype="int64") = optim_states[0]
            num_steps_new = R.call_tir(cls.add17, (num_steps,), out_sinfo=R.Tensor((), dtype="int64"))
            conv2d_weight: R.Tensor((64, 3, 3, 3), dtype="float32") = params[0]
            conv2d_weight_grad: R.Tensor((64, 3, 3, 3), dtype="float32") = gradients[0]
            conv2d_weight_v: R.Tensor((64, 3, 3, 3), dtype="float32") = optim_states[1]
            lv = R.call_tir(cls.multiply37, (conv2d_weight,), out_sinfo=R.Tensor((64, 3, 3, 3), dtype="float32"))
            conv2d_weight_grad_new = R.call_tir(cls.add18, (lv, conv2d_weight_grad), out_sinfo=R.Tensor((64, 3, 3, 3), dtype="float32"))
            lv1 = R.call_tir(cls.multiply38, (conv2d_weight_v,), out_sinfo=R.Tensor((64, 3, 3, 3), dtype="float32"))
            conv2d_weight_v_new = R.call_tir(cls.add18, (lv1, conv2d_weight_grad_new), out_sinfo=R.Tensor((64, 3, 3, 3), dtype="float32"))
            lv2 = R.call_tir(cls.multiply39, (conv2d_weight_v_new,), out_sinfo=R.Tensor((64, 3, 3, 3), dtype="float32"))
            conv2d_weight_new = R.call_tir(cls.subtract5, (conv2d_weight, lv2), out_sinfo=R.Tensor((64, 3, 3, 3), dtype="float32"))
            bn_gamma: R.Tensor((64,), dtype="float32") = params[1]
            bn_gamma_grad: R.Tensor((64,), dtype="float32") = gradients[1]
            bn_gamma_v: R.Tensor((64,), dtype="float32") = optim_states[2]
            lv3 = R.call_tir(cls.multiply40, (bn_gamma,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_gamma_grad_new = R.call_tir(cls.add13, (lv3, bn_gamma_grad), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv4 = R.call_tir(cls.multiply4, (bn_gamma_v,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_gamma_v_new = R.call_tir(cls.add13, (lv4, bn_gamma_grad_new), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv5 = R.call_tir(cls.multiply5, (bn_gamma_v_new,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_gamma_new = R.call_tir(cls.subtract6, (bn_gamma, lv5), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta: R.Tensor((64,), dtype="float32") = params[2]
            bn_beta_grad: R.Tensor((64,), dtype="float32") = gradients[2]
            bn_beta_v: R.Tensor((64,), dtype="float32") = optim_states[3]
            lv6 = R.call_tir(cls.multiply40, (bn_beta,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta_grad_new = R.call_tir(cls.add13, (lv6, bn_beta_grad), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv7 = R.call_tir(cls.multiply4, (bn_beta_v,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta_v_new = R.call_tir(cls.add13, (lv7, bn_beta_grad_new), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv8 = R.call_tir(cls.multiply5, (bn_beta_v_new,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta_new = R.call_tir(cls.subtract6, (bn_beta, lv8), out_sinfo=R.Tensor((64,), dtype="float32"))
            conv2d_weight1: R.Tensor((64, 64, 3, 3), dtype="float32") = params[3]
            conv2d_weight_grad1: R.Tensor((64, 64, 3, 3), dtype="float32") = gradients[3]
            conv2d_weight_v1: R.Tensor((64, 64, 3, 3), dtype="float32") = optim_states[4]
            lv9 = R.call_tir(cls.multiply41, (conv2d_weight1,), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            conv2d_weight_grad_new1 = R.call_tir(cls.add19, (lv9, conv2d_weight_grad1), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            lv10 = R.call_tir(cls.multiply42, (conv2d_weight_v1,), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            conv2d_weight_v_new1 = R.call_tir(cls.add19, (lv10, conv2d_weight_grad_new1), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            lv11 = R.call_tir(cls.multiply43, (conv2d_weight_v_new1,), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            conv2d_weight_new1 = R.call_tir(cls.subtract7, (conv2d_weight1, lv11), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            bn_gamma1: R.Tensor((64,), dtype="float32") = params[4]
            bn_gamma_grad1: R.Tensor((64,), dtype="float32") = gradients[4]
            bn_gamma_v1: R.Tensor((64,), dtype="float32") = optim_states[5]
            lv12 = R.call_tir(cls.multiply40, (bn_gamma1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_gamma_grad_new1 = R.call_tir(cls.add13, (lv12, bn_gamma_grad1), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv13 = R.call_tir(cls.multiply4, (bn_gamma_v1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_gamma_v_new1 = R.call_tir(cls.add13, (lv13, bn_gamma_grad_new1), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv14 = R.call_tir(cls.multiply5, (bn_gamma_v_new1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_gamma_new1 = R.call_tir(cls.subtract6, (bn_gamma1, lv14), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta1: R.Tensor((64,), dtype="float32") = params[5]
            bn_beta_grad1: R.Tensor((64,), dtype="float32") = gradients[5]
            bn_beta_v1: R.Tensor((64,), dtype="float32") = optim_states[6]
            lv15 = R.call_tir(cls.multiply40, (bn_beta1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta_grad_new1 = R.call_tir(cls.add13, (lv15, bn_beta_grad1), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv16 = R.call_tir(cls.multiply4, (bn_beta_v1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta_v_new1 = R.call_tir(cls.add13, (lv16, bn_beta_grad_new1), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv17 = R.call_tir(cls.multiply5, (bn_beta_v_new1,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta_new1 = R.call_tir(cls.subtract6, (bn_beta1, lv17), out_sinfo=R.Tensor((64,), dtype="float32"))
            conv2d_weight2: R.Tensor((64, 64, 3, 3), dtype="float32") = params[6]
            conv2d_weight_grad2: R.Tensor((64, 64, 3, 3), dtype="float32") = gradients[6]
            conv2d_weight_v2: R.Tensor((64, 64, 3, 3), dtype="float32") = optim_states[7]
            lv18 = R.call_tir(cls.multiply41, (conv2d_weight2,), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            conv2d_weight_grad_new2 = R.call_tir(cls.add19, (lv18, conv2d_weight_grad2), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            lv19 = R.call_tir(cls.multiply42, (conv2d_weight_v2,), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            conv2d_weight_v_new2 = R.call_tir(cls.add19, (lv19, conv2d_weight_grad_new2), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            lv20 = R.call_tir(cls.multiply43, (conv2d_weight_v_new2,), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            conv2d_weight_new2 = R.call_tir(cls.subtract7, (conv2d_weight2, lv20), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            bn_gamma2: R.Tensor((64,), dtype="float32") = params[7]
            bn_gamma_grad2: R.Tensor((64,), dtype="float32") = gradients[7]
            bn_gamma_v2: R.Tensor((64,), dtype="float32") = optim_states[8]
            lv21 = R.call_tir(cls.multiply40, (bn_gamma2,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_gamma_grad_new2 = R.call_tir(cls.add13, (lv21, bn_gamma_grad2), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv22 = R.call_tir(cls.multiply4, (bn_gamma_v2,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_gamma_v_new2 = R.call_tir(cls.add13, (lv22, bn_gamma_grad_new2), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv23 = R.call_tir(cls.multiply5, (bn_gamma_v_new2,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_gamma_new2 = R.call_tir(cls.subtract6, (bn_gamma2, lv23), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta2: R.Tensor((64,), dtype="float32") = params[8]
            bn_beta_grad2: R.Tensor((64,), dtype="float32") = gradients[8]
            bn_beta_v2: R.Tensor((64,), dtype="float32") = optim_states[9]
            lv24 = R.call_tir(cls.multiply40, (bn_beta2,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta_grad_new2 = R.call_tir(cls.add13, (lv24, bn_beta_grad2), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv25 = R.call_tir(cls.multiply4, (bn_beta_v2,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta_v_new2 = R.call_tir(cls.add13, (lv25, bn_beta_grad_new2), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv26 = R.call_tir(cls.multiply5, (bn_beta_v_new2,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta_new2 = R.call_tir(cls.subtract6, (bn_beta2, lv26), out_sinfo=R.Tensor((64,), dtype="float32"))
            conv2d_weight3: R.Tensor((64, 64, 3, 3), dtype="float32") = params[9]
            conv2d_weight_grad3: R.Tensor((64, 64, 3, 3), dtype="float32") = gradients[9]
            conv2d_weight_v3: R.Tensor((64, 64, 3, 3), dtype="float32") = optim_states[10]
            lv27 = R.call_tir(cls.multiply41, (conv2d_weight3,), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            conv2d_weight_grad_new3 = R.call_tir(cls.add19, (lv27, conv2d_weight_grad3), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            lv28 = R.call_tir(cls.multiply42, (conv2d_weight_v3,), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            conv2d_weight_v_new3 = R.call_tir(cls.add19, (lv28, conv2d_weight_grad_new3), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            lv29 = R.call_tir(cls.multiply43, (conv2d_weight_v_new3,), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            conv2d_weight_new3 = R.call_tir(cls.subtract7, (conv2d_weight3, lv29), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            bn_gamma3: R.Tensor((64,), dtype="float32") = params[10]
            bn_gamma_grad3: R.Tensor((64,), dtype="float32") = gradients[10]
            bn_gamma_v3: R.Tensor((64,), dtype="float32") = optim_states[11]
            lv30 = R.call_tir(cls.multiply40, (bn_gamma3,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_gamma_grad_new3 = R.call_tir(cls.add13, (lv30, bn_gamma_grad3), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv31 = R.call_tir(cls.multiply4, (bn_gamma_v3,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_gamma_v_new3 = R.call_tir(cls.add13, (lv31, bn_gamma_grad_new3), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv32 = R.call_tir(cls.multiply5, (bn_gamma_v_new3,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_gamma_new3 = R.call_tir(cls.subtract6, (bn_gamma3, lv32), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta3: R.Tensor((64,), dtype="float32") = params[11]
            bn_beta_grad3: R.Tensor((64,), dtype="float32") = gradients[11]
            bn_beta_v3: R.Tensor((64,), dtype="float32") = optim_states[12]
            lv33 = R.call_tir(cls.multiply40, (bn_beta3,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta_grad_new3 = R.call_tir(cls.add13, (lv33, bn_beta_grad3), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv34 = R.call_tir(cls.multiply4, (bn_beta_v3,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta_v_new3 = R.call_tir(cls.add13, (lv34, bn_beta_grad_new3), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv35 = R.call_tir(cls.multiply5, (bn_beta_v_new3,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta_new3 = R.call_tir(cls.subtract6, (bn_beta3, lv35), out_sinfo=R.Tensor((64,), dtype="float32"))
            conv2d_weight4: R.Tensor((64, 64, 3, 3), dtype="float32") = params[12]
            conv2d_weight_grad4: R.Tensor((64, 64, 3, 3), dtype="float32") = gradients[12]
            conv2d_weight_v4: R.Tensor((64, 64, 3, 3), dtype="float32") = optim_states[13]
            lv36 = R.call_tir(cls.multiply41, (conv2d_weight4,), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            conv2d_weight_grad_new4 = R.call_tir(cls.add19, (lv36, conv2d_weight_grad4), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            lv37 = R.call_tir(cls.multiply42, (conv2d_weight_v4,), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            conv2d_weight_v_new4 = R.call_tir(cls.add19, (lv37, conv2d_weight_grad_new4), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            lv38 = R.call_tir(cls.multiply43, (conv2d_weight_v_new4,), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            conv2d_weight_new4 = R.call_tir(cls.subtract7, (conv2d_weight4, lv38), out_sinfo=R.Tensor((64, 64, 3, 3), dtype="float32"))
            bn_gamma4: R.Tensor((64,), dtype="float32") = params[13]
            bn_gamma_grad4: R.Tensor((64,), dtype="float32") = gradients[13]
            bn_gamma_v4: R.Tensor((64,), dtype="float32") = optim_states[14]
            lv39 = R.call_tir(cls.multiply40, (bn_gamma4,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_gamma_grad_new4 = R.call_tir(cls.add13, (lv39, bn_gamma_grad4), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv40 = R.call_tir(cls.multiply4, (bn_gamma_v4,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_gamma_v_new4 = R.call_tir(cls.add13, (lv40, bn_gamma_grad_new4), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv41 = R.call_tir(cls.multiply5, (bn_gamma_v_new4,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_gamma_new4 = R.call_tir(cls.subtract6, (bn_gamma4, lv41), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta4: R.Tensor((64,), dtype="float32") = params[14]
            bn_beta_grad4: R.Tensor((64,), dtype="float32") = gradients[14]
            bn_beta_v4: R.Tensor((64,), dtype="float32") = optim_states[15]
            lv42 = R.call_tir(cls.multiply40, (bn_beta4,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta_grad_new4 = R.call_tir(cls.add13, (lv42, bn_beta_grad4), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv43 = R.call_tir(cls.multiply4, (bn_beta_v4,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta_v_new4 = R.call_tir(cls.add13, (lv43, bn_beta_grad_new4), out_sinfo=R.Tensor((64,), dtype="float32"))
            lv44 = R.call_tir(cls.multiply5, (bn_beta_v_new4,), out_sinfo=R.Tensor((64,), dtype="float32"))
            bn_beta_new4 = R.call_tir(cls.subtract6, (bn_beta4, lv44), out_sinfo=R.Tensor((64,), dtype="float32"))
            conv2d_weight5: R.Tensor((128, 64, 3, 3), dtype="float32") = params[15]
            conv2d_weight_grad5: R.Tensor((128, 64, 3, 3), dtype="float32") = gradients[15]
            conv2d_weight_v5: R.Tensor((128, 64, 3, 3), dtype="float32") = optim_states[16]
            lv45 = R.call_tir(cls.multiply44, (conv2d_weight5,), out_sinfo=R.Tensor((128, 64, 3, 3), dtype="float32"))
            conv2d_weight_grad_new5 = R.call_tir(cls.add20, (lv45, conv2d_weight_grad5), out_sinfo=R.Tensor((128, 64, 3, 3), dtype="float32"))
            lv46 = R.call_tir(cls.multiply45, (conv2d_weight_v5,), out_sinfo=R.Tensor((128, 64, 3, 3), dtype="float32"))
            conv2d_weight_v_new5 = R.call_tir(cls.add20, (lv46, conv2d_weight_grad_new5), out_sinfo=R.Tensor((128, 64, 3, 3), dtype="float32"))
            lv47 = R.call_tir(cls.multiply46, (conv2d_weight_v_new5,), out_sinfo=R.Tensor((128, 64, 3, 3), dtype="float32"))
            conv2d_weight_new5 = R.call_tir(cls.subtract8, (conv2d_weight5, lv47), out_sinfo=R.Tensor((128, 64, 3, 3), dtype="float32"))
            bn_gamma5: R.Tensor((128,), dtype="float32") = params[16]
            bn_gamma_grad5: R.Tensor((128,), dtype="float32") = gradients[16]
            bn_gamma_v5: R.Tensor((128,), dtype="float32") = optim_states[17]
            lv48 = R.call_tir(cls.multiply47, (bn_gamma5,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_gamma_grad_new5 = R.call_tir(cls.add14, (lv48, bn_gamma_grad5), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv49 = R.call_tir(cls.multiply6, (bn_gamma_v5,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_gamma_v_new5 = R.call_tir(cls.add14, (lv49, bn_gamma_grad_new5), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv50 = R.call_tir(cls.multiply7, (bn_gamma_v_new5,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_gamma_new5 = R.call_tir(cls.subtract9, (bn_gamma5, lv50), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta5: R.Tensor((128,), dtype="float32") = params[17]
            bn_beta_grad5: R.Tensor((128,), dtype="float32") = gradients[17]
            bn_beta_v5: R.Tensor((128,), dtype="float32") = optim_states[18]
            lv51 = R.call_tir(cls.multiply47, (bn_beta5,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta_grad_new5 = R.call_tir(cls.add14, (lv51, bn_beta_grad5), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv52 = R.call_tir(cls.multiply6, (bn_beta_v5,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta_v_new5 = R.call_tir(cls.add14, (lv52, bn_beta_grad_new5), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv53 = R.call_tir(cls.multiply7, (bn_beta_v_new5,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta_new5 = R.call_tir(cls.subtract9, (bn_beta5, lv53), out_sinfo=R.Tensor((128,), dtype="float32"))
            conv2d_weight6: R.Tensor((128, 128, 3, 3), dtype="float32") = params[18]
            conv2d_weight_grad6: R.Tensor((128, 128, 3, 3), dtype="float32") = gradients[18]
            conv2d_weight_v6: R.Tensor((128, 128, 3, 3), dtype="float32") = optim_states[19]
            lv54 = R.call_tir(cls.multiply48, (conv2d_weight6,), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            conv2d_weight_grad_new6 = R.call_tir(cls.add21, (lv54, conv2d_weight_grad6), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            lv55 = R.call_tir(cls.multiply49, (conv2d_weight_v6,), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            conv2d_weight_v_new6 = R.call_tir(cls.add21, (lv55, conv2d_weight_grad_new6), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            lv56 = R.call_tir(cls.multiply50, (conv2d_weight_v_new6,), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            conv2d_weight_new6 = R.call_tir(cls.subtract10, (conv2d_weight6, lv56), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            bn_gamma6: R.Tensor((128,), dtype="float32") = params[19]
            bn_gamma_grad6: R.Tensor((128,), dtype="float32") = gradients[19]
            bn_gamma_v6: R.Tensor((128,), dtype="float32") = optim_states[20]
            lv57 = R.call_tir(cls.multiply47, (bn_gamma6,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_gamma_grad_new6 = R.call_tir(cls.add14, (lv57, bn_gamma_grad6), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv58 = R.call_tir(cls.multiply6, (bn_gamma_v6,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_gamma_v_new6 = R.call_tir(cls.add14, (lv58, bn_gamma_grad_new6), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv59 = R.call_tir(cls.multiply7, (bn_gamma_v_new6,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_gamma_new6 = R.call_tir(cls.subtract9, (bn_gamma6, lv59), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta6: R.Tensor((128,), dtype="float32") = params[20]
            bn_beta_grad6: R.Tensor((128,), dtype="float32") = gradients[20]
            bn_beta_v6: R.Tensor((128,), dtype="float32") = optim_states[21]
            lv60 = R.call_tir(cls.multiply47, (bn_beta6,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta_grad_new6 = R.call_tir(cls.add14, (lv60, bn_beta_grad6), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv61 = R.call_tir(cls.multiply6, (bn_beta_v6,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta_v_new6 = R.call_tir(cls.add14, (lv61, bn_beta_grad_new6), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv62 = R.call_tir(cls.multiply7, (bn_beta_v_new6,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta_new6 = R.call_tir(cls.subtract9, (bn_beta6, lv62), out_sinfo=R.Tensor((128,), dtype="float32"))
            conv2d_weight7: R.Tensor((128, 64, 1, 1), dtype="float32") = params[21]
            conv2d_weight_grad7: R.Tensor((128, 64, 1, 1), dtype="float32") = gradients[21]
            conv2d_weight_v7: R.Tensor((128, 64, 1, 1), dtype="float32") = optim_states[22]
            lv63 = R.call_tir(cls.multiply51, (conv2d_weight7,), out_sinfo=R.Tensor((128, 64, 1, 1), dtype="float32"))
            conv2d_weight_grad_new7 = R.call_tir(cls.add22, (lv63, conv2d_weight_grad7), out_sinfo=R.Tensor((128, 64, 1, 1), dtype="float32"))
            lv64 = R.call_tir(cls.multiply52, (conv2d_weight_v7,), out_sinfo=R.Tensor((128, 64, 1, 1), dtype="float32"))
            conv2d_weight_v_new7 = R.call_tir(cls.add22, (lv64, conv2d_weight_grad_new7), out_sinfo=R.Tensor((128, 64, 1, 1), dtype="float32"))
            lv65 = R.call_tir(cls.multiply53, (conv2d_weight_v_new7,), out_sinfo=R.Tensor((128, 64, 1, 1), dtype="float32"))
            conv2d_weight_new7 = R.call_tir(cls.subtract11, (conv2d_weight7, lv65), out_sinfo=R.Tensor((128, 64, 1, 1), dtype="float32"))
            bn_gamma7: R.Tensor((128,), dtype="float32") = params[22]
            bn_gamma_grad7: R.Tensor((128,), dtype="float32") = gradients[22]
            bn_gamma_v7: R.Tensor((128,), dtype="float32") = optim_states[23]
            lv66 = R.call_tir(cls.multiply47, (bn_gamma7,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_gamma_grad_new7 = R.call_tir(cls.add14, (lv66, bn_gamma_grad7), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv67 = R.call_tir(cls.multiply6, (bn_gamma_v7,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_gamma_v_new7 = R.call_tir(cls.add14, (lv67, bn_gamma_grad_new7), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv68 = R.call_tir(cls.multiply7, (bn_gamma_v_new7,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_gamma_new7 = R.call_tir(cls.subtract9, (bn_gamma7, lv68), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta7: R.Tensor((128,), dtype="float32") = params[23]
            bn_beta_grad7: R.Tensor((128,), dtype="float32") = gradients[23]
            bn_beta_v7: R.Tensor((128,), dtype="float32") = optim_states[24]
            lv69 = R.call_tir(cls.multiply47, (bn_beta7,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta_grad_new7 = R.call_tir(cls.add14, (lv69, bn_beta_grad7), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv70 = R.call_tir(cls.multiply6, (bn_beta_v7,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta_v_new7 = R.call_tir(cls.add14, (lv70, bn_beta_grad_new7), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv71 = R.call_tir(cls.multiply7, (bn_beta_v_new7,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta_new7 = R.call_tir(cls.subtract9, (bn_beta7, lv71), out_sinfo=R.Tensor((128,), dtype="float32"))
            conv2d_weight8: R.Tensor((128, 128, 3, 3), dtype="float32") = params[24]
            conv2d_weight_grad8: R.Tensor((128, 128, 3, 3), dtype="float32") = gradients[24]
            conv2d_weight_v8: R.Tensor((128, 128, 3, 3), dtype="float32") = optim_states[25]
            lv72 = R.call_tir(cls.multiply48, (conv2d_weight8,), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            conv2d_weight_grad_new8 = R.call_tir(cls.add21, (lv72, conv2d_weight_grad8), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            lv73 = R.call_tir(cls.multiply49, (conv2d_weight_v8,), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            conv2d_weight_v_new8 = R.call_tir(cls.add21, (lv73, conv2d_weight_grad_new8), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            lv74 = R.call_tir(cls.multiply50, (conv2d_weight_v_new8,), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            conv2d_weight_new8 = R.call_tir(cls.subtract10, (conv2d_weight8, lv74), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            bn_gamma8: R.Tensor((128,), dtype="float32") = params[25]
            bn_gamma_grad8: R.Tensor((128,), dtype="float32") = gradients[25]
            bn_gamma_v8: R.Tensor((128,), dtype="float32") = optim_states[26]
            lv75 = R.call_tir(cls.multiply47, (bn_gamma8,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_gamma_grad_new8 = R.call_tir(cls.add14, (lv75, bn_gamma_grad8), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv76 = R.call_tir(cls.multiply6, (bn_gamma_v8,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_gamma_v_new8 = R.call_tir(cls.add14, (lv76, bn_gamma_grad_new8), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv77 = R.call_tir(cls.multiply7, (bn_gamma_v_new8,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_gamma_new8 = R.call_tir(cls.subtract9, (bn_gamma8, lv77), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta8: R.Tensor((128,), dtype="float32") = params[26]
            bn_beta_grad8: R.Tensor((128,), dtype="float32") = gradients[26]
            bn_beta_v8: R.Tensor((128,), dtype="float32") = optim_states[27]
            lv78 = R.call_tir(cls.multiply47, (bn_beta8,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta_grad_new8 = R.call_tir(cls.add14, (lv78, bn_beta_grad8), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv79 = R.call_tir(cls.multiply6, (bn_beta_v8,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta_v_new8 = R.call_tir(cls.add14, (lv79, bn_beta_grad_new8), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv80 = R.call_tir(cls.multiply7, (bn_beta_v_new8,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta_new8 = R.call_tir(cls.subtract9, (bn_beta8, lv80), out_sinfo=R.Tensor((128,), dtype="float32"))
            conv2d_weight9: R.Tensor((128, 128, 3, 3), dtype="float32") = params[27]
            conv2d_weight_grad9: R.Tensor((128, 128, 3, 3), dtype="float32") = gradients[27]
            conv2d_weight_v9: R.Tensor((128, 128, 3, 3), dtype="float32") = optim_states[28]
            lv81 = R.call_tir(cls.multiply48, (conv2d_weight9,), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            conv2d_weight_grad_new9 = R.call_tir(cls.add21, (lv81, conv2d_weight_grad9), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            lv82 = R.call_tir(cls.multiply49, (conv2d_weight_v9,), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            conv2d_weight_v_new9 = R.call_tir(cls.add21, (lv82, conv2d_weight_grad_new9), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            lv83 = R.call_tir(cls.multiply50, (conv2d_weight_v_new9,), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            conv2d_weight_new9 = R.call_tir(cls.subtract10, (conv2d_weight9, lv83), out_sinfo=R.Tensor((128, 128, 3, 3), dtype="float32"))
            bn_gamma9: R.Tensor((128,), dtype="float32") = params[28]
            bn_gamma_grad9: R.Tensor((128,), dtype="float32") = gradients[28]
            bn_gamma_v9: R.Tensor((128,), dtype="float32") = optim_states[29]
            lv84 = R.call_tir(cls.multiply47, (bn_gamma9,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_gamma_grad_new9 = R.call_tir(cls.add14, (lv84, bn_gamma_grad9), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv85 = R.call_tir(cls.multiply6, (bn_gamma_v9,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_gamma_v_new9 = R.call_tir(cls.add14, (lv85, bn_gamma_grad_new9), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv86 = R.call_tir(cls.multiply7, (bn_gamma_v_new9,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_gamma_new9 = R.call_tir(cls.subtract9, (bn_gamma9, lv86), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta9: R.Tensor((128,), dtype="float32") = params[29]
            bn_beta_grad9: R.Tensor((128,), dtype="float32") = gradients[29]
            bn_beta_v9: R.Tensor((128,), dtype="float32") = optim_states[30]
            lv87 = R.call_tir(cls.multiply47, (bn_beta9,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta_grad_new9 = R.call_tir(cls.add14, (lv87, bn_beta_grad9), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv88 = R.call_tir(cls.multiply6, (bn_beta_v9,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta_v_new9 = R.call_tir(cls.add14, (lv88, bn_beta_grad_new9), out_sinfo=R.Tensor((128,), dtype="float32"))
            lv89 = R.call_tir(cls.multiply7, (bn_beta_v_new9,), out_sinfo=R.Tensor((128,), dtype="float32"))
            bn_beta_new9 = R.call_tir(cls.subtract9, (bn_beta9, lv89), out_sinfo=R.Tensor((128,), dtype="float32"))
            conv2d_weight10: R.Tensor((256, 128, 3, 3), dtype="float32") = params[30]
            conv2d_weight_grad10: R.Tensor((256, 128, 3, 3), dtype="float32") = gradients[30]
            conv2d_weight_v10: R.Tensor((256, 128, 3, 3), dtype="float32") = optim_states[31]
            lv90 = R.call_tir(cls.multiply54, (conv2d_weight10,), out_sinfo=R.Tensor((256, 128, 3, 3), dtype="float32"))
            conv2d_weight_grad_new10 = R.call_tir(cls.add23, (lv90, conv2d_weight_grad10), out_sinfo=R.Tensor((256, 128, 3, 3), dtype="float32"))
            lv91 = R.call_tir(cls.multiply55, (conv2d_weight_v10,), out_sinfo=R.Tensor((256, 128, 3, 3), dtype="float32"))
            conv2d_weight_v_new10 = R.call_tir(cls.add23, (lv91, conv2d_weight_grad_new10), out_sinfo=R.Tensor((256, 128, 3, 3), dtype="float32"))
            lv92 = R.call_tir(cls.multiply56, (conv2d_weight_v_new10,), out_sinfo=R.Tensor((256, 128, 3, 3), dtype="float32"))
            conv2d_weight_new10 = R.call_tir(cls.subtract12, (conv2d_weight10, lv92), out_sinfo=R.Tensor((256, 128, 3, 3), dtype="float32"))
            bn_gamma10: R.Tensor((256,), dtype="float32") = params[31]
            bn_gamma_grad10: R.Tensor((256,), dtype="float32") = gradients[31]
            bn_gamma_v10: R.Tensor((256,), dtype="float32") = optim_states[32]
            lv93 = R.call_tir(cls.multiply57, (bn_gamma10,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_gamma_grad_new10 = R.call_tir(cls.add15, (lv93, bn_gamma_grad10), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv94 = R.call_tir(cls.multiply8, (bn_gamma_v10,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_gamma_v_new10 = R.call_tir(cls.add15, (lv94, bn_gamma_grad_new10), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv95 = R.call_tir(cls.multiply9, (bn_gamma_v_new10,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_gamma_new10 = R.call_tir(cls.subtract13, (bn_gamma10, lv95), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta10: R.Tensor((256,), dtype="float32") = params[32]
            bn_beta_grad10: R.Tensor((256,), dtype="float32") = gradients[32]
            bn_beta_v10: R.Tensor((256,), dtype="float32") = optim_states[33]
            lv96 = R.call_tir(cls.multiply57, (bn_beta10,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta_grad_new10 = R.call_tir(cls.add15, (lv96, bn_beta_grad10), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv97 = R.call_tir(cls.multiply8, (bn_beta_v10,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta_v_new10 = R.call_tir(cls.add15, (lv97, bn_beta_grad_new10), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv98 = R.call_tir(cls.multiply9, (bn_beta_v_new10,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta_new10 = R.call_tir(cls.subtract13, (bn_beta10, lv98), out_sinfo=R.Tensor((256,), dtype="float32"))
            conv2d_weight11: R.Tensor((256, 256, 3, 3), dtype="float32") = params[33]
            conv2d_weight_grad11: R.Tensor((256, 256, 3, 3), dtype="float32") = gradients[33]
            conv2d_weight_v11: R.Tensor((256, 256, 3, 3), dtype="float32") = optim_states[34]
            lv99 = R.call_tir(cls.multiply58, (conv2d_weight11,), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            conv2d_weight_grad_new11 = R.call_tir(cls.add24, (lv99, conv2d_weight_grad11), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            lv100 = R.call_tir(cls.multiply59, (conv2d_weight_v11,), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            conv2d_weight_v_new11 = R.call_tir(cls.add24, (lv100, conv2d_weight_grad_new11), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            lv101 = R.call_tir(cls.multiply60, (conv2d_weight_v_new11,), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            conv2d_weight_new11 = R.call_tir(cls.subtract14, (conv2d_weight11, lv101), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            bn_gamma11: R.Tensor((256,), dtype="float32") = params[34]
            bn_gamma_grad11: R.Tensor((256,), dtype="float32") = gradients[34]
            bn_gamma_v11: R.Tensor((256,), dtype="float32") = optim_states[35]
            lv102 = R.call_tir(cls.multiply57, (bn_gamma11,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_gamma_grad_new11 = R.call_tir(cls.add15, (lv102, bn_gamma_grad11), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv103 = R.call_tir(cls.multiply8, (bn_gamma_v11,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_gamma_v_new11 = R.call_tir(cls.add15, (lv103, bn_gamma_grad_new11), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv104 = R.call_tir(cls.multiply9, (bn_gamma_v_new11,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_gamma_new11 = R.call_tir(cls.subtract13, (bn_gamma11, lv104), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta11: R.Tensor((256,), dtype="float32") = params[35]
            bn_beta_grad11: R.Tensor((256,), dtype="float32") = gradients[35]
            bn_beta_v11: R.Tensor((256,), dtype="float32") = optim_states[36]
            lv105 = R.call_tir(cls.multiply57, (bn_beta11,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta_grad_new11 = R.call_tir(cls.add15, (lv105, bn_beta_grad11), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv106 = R.call_tir(cls.multiply8, (bn_beta_v11,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta_v_new11 = R.call_tir(cls.add15, (lv106, bn_beta_grad_new11), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv107 = R.call_tir(cls.multiply9, (bn_beta_v_new11,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta_new11 = R.call_tir(cls.subtract13, (bn_beta11, lv107), out_sinfo=R.Tensor((256,), dtype="float32"))
            conv2d_weight12: R.Tensor((256, 128, 1, 1), dtype="float32") = params[36]
            conv2d_weight_grad12: R.Tensor((256, 128, 1, 1), dtype="float32") = gradients[36]
            conv2d_weight_v12: R.Tensor((256, 128, 1, 1), dtype="float32") = optim_states[37]
            lv108 = R.call_tir(cls.multiply61, (conv2d_weight12,), out_sinfo=R.Tensor((256, 128, 1, 1), dtype="float32"))
            conv2d_weight_grad_new12 = R.call_tir(cls.add25, (lv108, conv2d_weight_grad12), out_sinfo=R.Tensor((256, 128, 1, 1), dtype="float32"))
            lv109 = R.call_tir(cls.multiply62, (conv2d_weight_v12,), out_sinfo=R.Tensor((256, 128, 1, 1), dtype="float32"))
            conv2d_weight_v_new12 = R.call_tir(cls.add25, (lv109, conv2d_weight_grad_new12), out_sinfo=R.Tensor((256, 128, 1, 1), dtype="float32"))
            lv110 = R.call_tir(cls.multiply63, (conv2d_weight_v_new12,), out_sinfo=R.Tensor((256, 128, 1, 1), dtype="float32"))
            conv2d_weight_new12 = R.call_tir(cls.subtract15, (conv2d_weight12, lv110), out_sinfo=R.Tensor((256, 128, 1, 1), dtype="float32"))
            bn_gamma12: R.Tensor((256,), dtype="float32") = params[37]
            bn_gamma_grad12: R.Tensor((256,), dtype="float32") = gradients[37]
            bn_gamma_v12: R.Tensor((256,), dtype="float32") = optim_states[38]
            lv111 = R.call_tir(cls.multiply57, (bn_gamma12,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_gamma_grad_new12 = R.call_tir(cls.add15, (lv111, bn_gamma_grad12), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv112 = R.call_tir(cls.multiply8, (bn_gamma_v12,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_gamma_v_new12 = R.call_tir(cls.add15, (lv112, bn_gamma_grad_new12), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv113 = R.call_tir(cls.multiply9, (bn_gamma_v_new12,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_gamma_new12 = R.call_tir(cls.subtract13, (bn_gamma12, lv113), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta12: R.Tensor((256,), dtype="float32") = params[38]
            bn_beta_grad12: R.Tensor((256,), dtype="float32") = gradients[38]
            bn_beta_v12: R.Tensor((256,), dtype="float32") = optim_states[39]
            lv114 = R.call_tir(cls.multiply57, (bn_beta12,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta_grad_new12 = R.call_tir(cls.add15, (lv114, bn_beta_grad12), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv115 = R.call_tir(cls.multiply8, (bn_beta_v12,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta_v_new12 = R.call_tir(cls.add15, (lv115, bn_beta_grad_new12), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv116 = R.call_tir(cls.multiply9, (bn_beta_v_new12,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta_new12 = R.call_tir(cls.subtract13, (bn_beta12, lv116), out_sinfo=R.Tensor((256,), dtype="float32"))
            conv2d_weight13: R.Tensor((256, 256, 3, 3), dtype="float32") = params[39]
            conv2d_weight_grad13: R.Tensor((256, 256, 3, 3), dtype="float32") = gradients[39]
            conv2d_weight_v13: R.Tensor((256, 256, 3, 3), dtype="float32") = optim_states[40]
            lv117 = R.call_tir(cls.multiply58, (conv2d_weight13,), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            conv2d_weight_grad_new13 = R.call_tir(cls.add24, (lv117, conv2d_weight_grad13), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            lv118 = R.call_tir(cls.multiply59, (conv2d_weight_v13,), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            conv2d_weight_v_new13 = R.call_tir(cls.add24, (lv118, conv2d_weight_grad_new13), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            lv119 = R.call_tir(cls.multiply60, (conv2d_weight_v_new13,), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            conv2d_weight_new13 = R.call_tir(cls.subtract14, (conv2d_weight13, lv119), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            bn_gamma13: R.Tensor((256,), dtype="float32") = params[40]
            bn_gamma_grad13: R.Tensor((256,), dtype="float32") = gradients[40]
            bn_gamma_v13: R.Tensor((256,), dtype="float32") = optim_states[41]
            lv120 = R.call_tir(cls.multiply57, (bn_gamma13,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_gamma_grad_new13 = R.call_tir(cls.add15, (lv120, bn_gamma_grad13), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv121 = R.call_tir(cls.multiply8, (bn_gamma_v13,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_gamma_v_new13 = R.call_tir(cls.add15, (lv121, bn_gamma_grad_new13), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv122 = R.call_tir(cls.multiply9, (bn_gamma_v_new13,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_gamma_new13 = R.call_tir(cls.subtract13, (bn_gamma13, lv122), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta13: R.Tensor((256,), dtype="float32") = params[41]
            bn_beta_grad13: R.Tensor((256,), dtype="float32") = gradients[41]
            bn_beta_v13: R.Tensor((256,), dtype="float32") = optim_states[42]
            lv123 = R.call_tir(cls.multiply57, (bn_beta13,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta_grad_new13 = R.call_tir(cls.add15, (lv123, bn_beta_grad13), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv124 = R.call_tir(cls.multiply8, (bn_beta_v13,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta_v_new13 = R.call_tir(cls.add15, (lv124, bn_beta_grad_new13), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv125 = R.call_tir(cls.multiply9, (bn_beta_v_new13,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta_new13 = R.call_tir(cls.subtract13, (bn_beta13, lv125), out_sinfo=R.Tensor((256,), dtype="float32"))
            conv2d_weight14: R.Tensor((256, 256, 3, 3), dtype="float32") = params[42]
            conv2d_weight_grad14: R.Tensor((256, 256, 3, 3), dtype="float32") = gradients[42]
            conv2d_weight_v14: R.Tensor((256, 256, 3, 3), dtype="float32") = optim_states[43]
            lv126 = R.call_tir(cls.multiply58, (conv2d_weight14,), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            conv2d_weight_grad_new14 = R.call_tir(cls.add24, (lv126, conv2d_weight_grad14), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            lv127 = R.call_tir(cls.multiply59, (conv2d_weight_v14,), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            conv2d_weight_v_new14 = R.call_tir(cls.add24, (lv127, conv2d_weight_grad_new14), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            lv128 = R.call_tir(cls.multiply60, (conv2d_weight_v_new14,), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            conv2d_weight_new14 = R.call_tir(cls.subtract14, (conv2d_weight14, lv128), out_sinfo=R.Tensor((256, 256, 3, 3), dtype="float32"))
            bn_gamma14: R.Tensor((256,), dtype="float32") = params[43]
            bn_gamma_grad14: R.Tensor((256,), dtype="float32") = gradients[43]
            bn_gamma_v14: R.Tensor((256,), dtype="float32") = optim_states[44]
            lv129 = R.call_tir(cls.multiply57, (bn_gamma14,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_gamma_grad_new14 = R.call_tir(cls.add15, (lv129, bn_gamma_grad14), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv130 = R.call_tir(cls.multiply8, (bn_gamma_v14,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_gamma_v_new14 = R.call_tir(cls.add15, (lv130, bn_gamma_grad_new14), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv131 = R.call_tir(cls.multiply9, (bn_gamma_v_new14,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_gamma_new14 = R.call_tir(cls.subtract13, (bn_gamma14, lv131), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta14: R.Tensor((256,), dtype="float32") = params[44]
            bn_beta_grad14: R.Tensor((256,), dtype="float32") = gradients[44]
            bn_beta_v14: R.Tensor((256,), dtype="float32") = optim_states[45]
            lv132 = R.call_tir(cls.multiply57, (bn_beta14,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta_grad_new14 = R.call_tir(cls.add15, (lv132, bn_beta_grad14), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv133 = R.call_tir(cls.multiply8, (bn_beta_v14,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta_v_new14 = R.call_tir(cls.add15, (lv133, bn_beta_grad_new14), out_sinfo=R.Tensor((256,), dtype="float32"))
            lv134 = R.call_tir(cls.multiply9, (bn_beta_v_new14,), out_sinfo=R.Tensor((256,), dtype="float32"))
            bn_beta_new14 = R.call_tir(cls.subtract13, (bn_beta14, lv134), out_sinfo=R.Tensor((256,), dtype="float32"))
            conv2d_weight15: R.Tensor((512, 256, 3, 3), dtype="float32") = params[45]
            conv2d_weight_grad15: R.Tensor((512, 256, 3, 3), dtype="float32") = gradients[45]
            conv2d_weight_v15: R.Tensor((512, 256, 3, 3), dtype="float32") = optim_states[46]
            lv135 = R.call_tir(cls.multiply64, (conv2d_weight15,), out_sinfo=R.Tensor((512, 256, 3, 3), dtype="float32"))
            conv2d_weight_grad_new15 = R.call_tir(cls.add26, (lv135, conv2d_weight_grad15), out_sinfo=R.Tensor((512, 256, 3, 3), dtype="float32"))
            lv136 = R.call_tir(cls.multiply65, (conv2d_weight_v15,), out_sinfo=R.Tensor((512, 256, 3, 3), dtype="float32"))
            conv2d_weight_v_new15 = R.call_tir(cls.add26, (lv136, conv2d_weight_grad_new15), out_sinfo=R.Tensor((512, 256, 3, 3), dtype="float32"))
            lv137 = R.call_tir(cls.multiply66, (conv2d_weight_v_new15,), out_sinfo=R.Tensor((512, 256, 3, 3), dtype="float32"))
            conv2d_weight_new15 = R.call_tir(cls.subtract16, (conv2d_weight15, lv137), out_sinfo=R.Tensor((512, 256, 3, 3), dtype="float32"))
            bn_gamma15: R.Tensor((512,), dtype="float32") = params[46]
            bn_gamma_grad15: R.Tensor((512,), dtype="float32") = gradients[46]
            bn_gamma_v15: R.Tensor((512,), dtype="float32") = optim_states[47]
            lv138 = R.call_tir(cls.multiply67, (bn_gamma15,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_gamma_grad_new15 = R.call_tir(cls.add16, (lv138, bn_gamma_grad15), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv139 = R.call_tir(cls.multiply10, (bn_gamma_v15,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_gamma_v_new15 = R.call_tir(cls.add16, (lv139, bn_gamma_grad_new15), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv140 = R.call_tir(cls.multiply11, (bn_gamma_v_new15,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_gamma_new15 = R.call_tir(cls.subtract17, (bn_gamma15, lv140), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta15: R.Tensor((512,), dtype="float32") = params[47]
            bn_beta_grad15: R.Tensor((512,), dtype="float32") = gradients[47]
            bn_beta_v15: R.Tensor((512,), dtype="float32") = optim_states[48]
            lv141 = R.call_tir(cls.multiply67, (bn_beta15,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta_grad_new15 = R.call_tir(cls.add16, (lv141, bn_beta_grad15), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv142 = R.call_tir(cls.multiply10, (bn_beta_v15,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta_v_new15 = R.call_tir(cls.add16, (lv142, bn_beta_grad_new15), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv143 = R.call_tir(cls.multiply11, (bn_beta_v_new15,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta_new15 = R.call_tir(cls.subtract17, (bn_beta15, lv143), out_sinfo=R.Tensor((512,), dtype="float32"))
            conv2d_weight16: R.Tensor((512, 512, 3, 3), dtype="float32") = params[48]
            conv2d_weight_grad16: R.Tensor((512, 512, 3, 3), dtype="float32") = gradients[48]
            conv2d_weight_v16: R.Tensor((512, 512, 3, 3), dtype="float32") = optim_states[49]
            lv144 = R.call_tir(cls.multiply68, (conv2d_weight16,), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            conv2d_weight_grad_new16 = R.call_tir(cls.add27, (lv144, conv2d_weight_grad16), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            lv145 = R.call_tir(cls.multiply69, (conv2d_weight_v16,), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            conv2d_weight_v_new16 = R.call_tir(cls.add27, (lv145, conv2d_weight_grad_new16), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            lv146 = R.call_tir(cls.multiply70, (conv2d_weight_v_new16,), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            conv2d_weight_new16 = R.call_tir(cls.subtract18, (conv2d_weight16, lv146), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            bn_gamma16: R.Tensor((512,), dtype="float32") = params[49]
            bn_gamma_grad16: R.Tensor((512,), dtype="float32") = gradients[49]
            bn_gamma_v16: R.Tensor((512,), dtype="float32") = optim_states[50]
            lv147 = R.call_tir(cls.multiply67, (bn_gamma16,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_gamma_grad_new16 = R.call_tir(cls.add16, (lv147, bn_gamma_grad16), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv148 = R.call_tir(cls.multiply10, (bn_gamma_v16,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_gamma_v_new16 = R.call_tir(cls.add16, (lv148, bn_gamma_grad_new16), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv149 = R.call_tir(cls.multiply11, (bn_gamma_v_new16,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_gamma_new16 = R.call_tir(cls.subtract17, (bn_gamma16, lv149), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta16: R.Tensor((512,), dtype="float32") = params[50]
            bn_beta_grad16: R.Tensor((512,), dtype="float32") = gradients[50]
            bn_beta_v16: R.Tensor((512,), dtype="float32") = optim_states[51]
            lv150 = R.call_tir(cls.multiply67, (bn_beta16,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta_grad_new16 = R.call_tir(cls.add16, (lv150, bn_beta_grad16), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv151 = R.call_tir(cls.multiply10, (bn_beta_v16,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta_v_new16 = R.call_tir(cls.add16, (lv151, bn_beta_grad_new16), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv152 = R.call_tir(cls.multiply11, (bn_beta_v_new16,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta_new16 = R.call_tir(cls.subtract17, (bn_beta16, lv152), out_sinfo=R.Tensor((512,), dtype="float32"))
            conv2d_weight17: R.Tensor((512, 256, 1, 1), dtype="float32") = params[51]
            conv2d_weight_grad17: R.Tensor((512, 256, 1, 1), dtype="float32") = gradients[51]
            conv2d_weight_v17: R.Tensor((512, 256, 1, 1), dtype="float32") = optim_states[52]
            lv153 = R.call_tir(cls.multiply71, (conv2d_weight17,), out_sinfo=R.Tensor((512, 256, 1, 1), dtype="float32"))
            conv2d_weight_grad_new17 = R.call_tir(cls.add28, (lv153, conv2d_weight_grad17), out_sinfo=R.Tensor((512, 256, 1, 1), dtype="float32"))
            lv154 = R.call_tir(cls.multiply72, (conv2d_weight_v17,), out_sinfo=R.Tensor((512, 256, 1, 1), dtype="float32"))
            conv2d_weight_v_new17 = R.call_tir(cls.add28, (lv154, conv2d_weight_grad_new17), out_sinfo=R.Tensor((512, 256, 1, 1), dtype="float32"))
            lv155 = R.call_tir(cls.multiply73, (conv2d_weight_v_new17,), out_sinfo=R.Tensor((512, 256, 1, 1), dtype="float32"))
            conv2d_weight_new17 = R.call_tir(cls.subtract19, (conv2d_weight17, lv155), out_sinfo=R.Tensor((512, 256, 1, 1), dtype="float32"))
            bn_gamma17: R.Tensor((512,), dtype="float32") = params[52]
            bn_gamma_grad17: R.Tensor((512,), dtype="float32") = gradients[52]
            bn_gamma_v17: R.Tensor((512,), dtype="float32") = optim_states[53]
            lv156 = R.call_tir(cls.multiply67, (bn_gamma17,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_gamma_grad_new17 = R.call_tir(cls.add16, (lv156, bn_gamma_grad17), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv157 = R.call_tir(cls.multiply10, (bn_gamma_v17,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_gamma_v_new17 = R.call_tir(cls.add16, (lv157, bn_gamma_grad_new17), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv158 = R.call_tir(cls.multiply11, (bn_gamma_v_new17,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_gamma_new17 = R.call_tir(cls.subtract17, (bn_gamma17, lv158), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta17: R.Tensor((512,), dtype="float32") = params[53]
            bn_beta_grad17: R.Tensor((512,), dtype="float32") = gradients[53]
            bn_beta_v17: R.Tensor((512,), dtype="float32") = optim_states[54]
            lv159 = R.call_tir(cls.multiply67, (bn_beta17,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta_grad_new17 = R.call_tir(cls.add16, (lv159, bn_beta_grad17), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv160 = R.call_tir(cls.multiply10, (bn_beta_v17,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta_v_new17 = R.call_tir(cls.add16, (lv160, bn_beta_grad_new17), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv161 = R.call_tir(cls.multiply11, (bn_beta_v_new17,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta_new17 = R.call_tir(cls.subtract17, (bn_beta17, lv161), out_sinfo=R.Tensor((512,), dtype="float32"))
            conv2d_weight18: R.Tensor((512, 512, 3, 3), dtype="float32") = params[54]
            conv2d_weight_grad18: R.Tensor((512, 512, 3, 3), dtype="float32") = gradients[54]
            conv2d_weight_v18: R.Tensor((512, 512, 3, 3), dtype="float32") = optim_states[55]
            lv162 = R.call_tir(cls.multiply68, (conv2d_weight18,), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            conv2d_weight_grad_new18 = R.call_tir(cls.add27, (lv162, conv2d_weight_grad18), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            lv163 = R.call_tir(cls.multiply69, (conv2d_weight_v18,), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            conv2d_weight_v_new18 = R.call_tir(cls.add27, (lv163, conv2d_weight_grad_new18), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            lv164 = R.call_tir(cls.multiply70, (conv2d_weight_v_new18,), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            conv2d_weight_new18 = R.call_tir(cls.subtract18, (conv2d_weight18, lv164), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            bn_gamma18: R.Tensor((512,), dtype="float32") = params[55]
            bn_gamma_grad18: R.Tensor((512,), dtype="float32") = gradients[55]
            bn_gamma_v18: R.Tensor((512,), dtype="float32") = optim_states[56]
            lv165 = R.call_tir(cls.multiply67, (bn_gamma18,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_gamma_grad_new18 = R.call_tir(cls.add16, (lv165, bn_gamma_grad18), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv166 = R.call_tir(cls.multiply10, (bn_gamma_v18,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_gamma_v_new18 = R.call_tir(cls.add16, (lv166, bn_gamma_grad_new18), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv167 = R.call_tir(cls.multiply11, (bn_gamma_v_new18,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_gamma_new18 = R.call_tir(cls.subtract17, (bn_gamma18, lv167), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta18: R.Tensor((512,), dtype="float32") = params[56]
            bn_beta_grad18: R.Tensor((512,), dtype="float32") = gradients[56]
            bn_beta_v18: R.Tensor((512,), dtype="float32") = optim_states[57]
            lv168 = R.call_tir(cls.multiply67, (bn_beta18,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta_grad_new18 = R.call_tir(cls.add16, (lv168, bn_beta_grad18), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv169 = R.call_tir(cls.multiply10, (bn_beta_v18,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta_v_new18 = R.call_tir(cls.add16, (lv169, bn_beta_grad_new18), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv170 = R.call_tir(cls.multiply11, (bn_beta_v_new18,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta_new18 = R.call_tir(cls.subtract17, (bn_beta18, lv170), out_sinfo=R.Tensor((512,), dtype="float32"))
            conv2d_weight19: R.Tensor((512, 512, 3, 3), dtype="float32") = params[57]
            conv2d_weight_grad19: R.Tensor((512, 512, 3, 3), dtype="float32") = gradients[57]
            conv2d_weight_v19: R.Tensor((512, 512, 3, 3), dtype="float32") = optim_states[58]
            lv171 = R.call_tir(cls.multiply68, (conv2d_weight19,), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            conv2d_weight_grad_new19 = R.call_tir(cls.add27, (lv171, conv2d_weight_grad19), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            lv172 = R.call_tir(cls.multiply69, (conv2d_weight_v19,), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            conv2d_weight_v_new19 = R.call_tir(cls.add27, (lv172, conv2d_weight_grad_new19), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            lv173 = R.call_tir(cls.multiply70, (conv2d_weight_v_new19,), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            conv2d_weight_new19 = R.call_tir(cls.subtract18, (conv2d_weight19, lv173), out_sinfo=R.Tensor((512, 512, 3, 3), dtype="float32"))
            bn_gamma19: R.Tensor((512,), dtype="float32") = params[58]
            bn_gamma_grad19: R.Tensor((512,), dtype="float32") = gradients[58]
            bn_gamma_v19: R.Tensor((512,), dtype="float32") = optim_states[59]
            lv174 = R.call_tir(cls.multiply67, (bn_gamma19,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_gamma_grad_new19 = R.call_tir(cls.add16, (lv174, bn_gamma_grad19), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv175 = R.call_tir(cls.multiply10, (bn_gamma_v19,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_gamma_v_new19 = R.call_tir(cls.add16, (lv175, bn_gamma_grad_new19), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv176 = R.call_tir(cls.multiply11, (bn_gamma_v_new19,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_gamma_new19 = R.call_tir(cls.subtract17, (bn_gamma19, lv176), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta19: R.Tensor((512,), dtype="float32") = params[59]
            bn_beta_grad19: R.Tensor((512,), dtype="float32") = gradients[59]
            bn_beta_v19: R.Tensor((512,), dtype="float32") = optim_states[60]
            lv177 = R.call_tir(cls.multiply67, (bn_beta19,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta_grad_new19 = R.call_tir(cls.add16, (lv177, bn_beta_grad19), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv178 = R.call_tir(cls.multiply10, (bn_beta_v19,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta_v_new19 = R.call_tir(cls.add16, (lv178, bn_beta_grad_new19), out_sinfo=R.Tensor((512,), dtype="float32"))
            lv179 = R.call_tir(cls.multiply11, (bn_beta_v_new19,), out_sinfo=R.Tensor((512,), dtype="float32"))
            bn_beta_new19 = R.call_tir(cls.subtract17, (bn_beta19, lv179), out_sinfo=R.Tensor((512,), dtype="float32"))
            ln_weight: R.Tensor((512, 10), dtype="float32") = params[60]
            ln_weight_grad: R.Tensor((512, 10), dtype="float32") = gradients[60]
            ln_weight_v: R.Tensor((512, 10), dtype="float32") = optim_states[61]
            lv180 = R.call_tir(cls.multiply74, (ln_weight,), out_sinfo=R.Tensor((512, 10), dtype="float32"))
            ln_weight_grad_new = R.call_tir(cls.add29, (lv180, ln_weight_grad), out_sinfo=R.Tensor((512, 10), dtype="float32"))
            lv181 = R.call_tir(cls.multiply75, (ln_weight_v,), out_sinfo=R.Tensor((512, 10), dtype="float32"))
            ln_weight_v_new = R.call_tir(cls.add29, (lv181, ln_weight_grad_new), out_sinfo=R.Tensor((512, 10), dtype="float32"))
            lv182 = R.call_tir(cls.multiply76, (ln_weight_v_new,), out_sinfo=R.Tensor((512, 10), dtype="float32"))
            ln_weight_new = R.call_tir(cls.subtract20, (ln_weight, lv182), out_sinfo=R.Tensor((512, 10), dtype="float32"))
            ln_bias: R.Tensor((10,), dtype="float32") = params[61]
            ln_bias_grad: R.Tensor((10,), dtype="float32") = gradients[61]
            ln_bias_v: R.Tensor((10,), dtype="float32") = optim_states[62]
            lv183 = R.call_tir(cls.multiply77, (ln_bias,), out_sinfo=R.Tensor((10,), dtype="float32"))
            ln_bias_grad_new = R.call_tir(cls.add30, (lv183, ln_bias_grad), out_sinfo=R.Tensor((10,), dtype="float32"))
            lv184 = R.call_tir(cls.multiply78, (ln_bias_v,), out_sinfo=R.Tensor((10,), dtype="float32"))
            ln_bias_v_new = R.call_tir(cls.add30, (lv184, ln_bias_grad_new), out_sinfo=R.Tensor((10,), dtype="float32"))
            lv185 = R.call_tir(cls.multiply79, (ln_bias_v_new,), out_sinfo=R.Tensor((10,), dtype="float32"))
            ln_bias_new = R.call_tir(cls.subtract21, (ln_bias, lv185), out_sinfo=R.Tensor((10,), dtype="float32"))
            params_new: R.Tuple(R.Tensor((64, 3, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((128, 64, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 64, 1, 1), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((256, 128, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 128, 1, 1), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((512, 256, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 256, 1, 1), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 10), dtype="float32"), R.Tensor((10,), dtype="float32")) = conv2d_weight_new, bn_gamma_new, bn_beta_new, conv2d_weight_new1, bn_gamma_new1, bn_beta_new1, conv2d_weight_new2, bn_gamma_new2, bn_beta_new2, conv2d_weight_new3, bn_gamma_new3, bn_beta_new3, conv2d_weight_new4, bn_gamma_new4, bn_beta_new4, conv2d_weight_new5, bn_gamma_new5, bn_beta_new5, conv2d_weight_new6, bn_gamma_new6, bn_beta_new6, conv2d_weight_new7, bn_gamma_new7, bn_beta_new7, conv2d_weight_new8, bn_gamma_new8, bn_beta_new8, conv2d_weight_new9, bn_gamma_new9, bn_beta_new9, conv2d_weight_new10, bn_gamma_new10, bn_beta_new10, conv2d_weight_new11, bn_gamma_new11, bn_beta_new11, conv2d_weight_new12, bn_gamma_new12, bn_beta_new12, conv2d_weight_new13, bn_gamma_new13, bn_beta_new13, conv2d_weight_new14, bn_gamma_new14, bn_beta_new14, conv2d_weight_new15, bn_gamma_new15, bn_beta_new15, conv2d_weight_new16, bn_gamma_new16, bn_beta_new16, conv2d_weight_new17, bn_gamma_new17, bn_beta_new17, conv2d_weight_new18, bn_gamma_new18, bn_beta_new18, conv2d_weight_new19, bn_gamma_new19, bn_beta_new19, ln_weight_new, ln_bias_new
            optim_states_new: R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((64, 3, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64, 64, 3, 3), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((128, 64, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 64, 1, 1), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128, 128, 3, 3), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((256, 128, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 128, 1, 1), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256, 256, 3, 3), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((512, 256, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 256, 1, 1), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 512, 3, 3), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512, 10), dtype="float32"), R.Tensor((10,), dtype="float32")) = num_steps_new, conv2d_weight_v_new, bn_gamma_v_new, bn_beta_v_new, conv2d_weight_v_new1, bn_gamma_v_new1, bn_beta_v_new1, conv2d_weight_v_new2, bn_gamma_v_new2, bn_beta_v_new2, conv2d_weight_v_new3, bn_gamma_v_new3, bn_beta_v_new3, conv2d_weight_v_new4, bn_gamma_v_new4, bn_beta_v_new4, conv2d_weight_v_new5, bn_gamma_v_new5, bn_beta_v_new5, conv2d_weight_v_new6, bn_gamma_v_new6, bn_beta_v_new6, conv2d_weight_v_new7, bn_gamma_v_new7, bn_beta_v_new7, conv2d_weight_v_new8, bn_gamma_v_new8, bn_beta_v_new8, conv2d_weight_v_new9, bn_gamma_v_new9, bn_beta_v_new9, conv2d_weight_v_new10, bn_gamma_v_new10, bn_beta_v_new10, conv2d_weight_v_new11, bn_gamma_v_new11, bn_beta_v_new11, conv2d_weight_v_new12, bn_gamma_v_new12, bn_beta_v_new12, conv2d_weight_v_new13, bn_gamma_v_new13, bn_beta_v_new13, conv2d_weight_v_new14, bn_gamma_v_new14, bn_beta_v_new14, conv2d_weight_v_new15, bn_gamma_v_new15, bn_beta_v_new15, conv2d_weight_v_new16, bn_gamma_v_new16, bn_beta_v_new16, conv2d_weight_v_new17, bn_gamma_v_new17, bn_beta_v_new17, conv2d_weight_v_new18, bn_gamma_v_new18, bn_beta_v_new18, conv2d_weight_v_new19, bn_gamma_v_new19, bn_beta_v_new19, ln_weight_v_new, ln_bias_v_new
            R.output(params_new, optim_states_new)
        return (params_new, optim_states_new)
