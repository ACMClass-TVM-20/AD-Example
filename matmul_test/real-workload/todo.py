from tvm.script import tir as T


@T.prim_func
def fused_fused_decode4_fused_NT_matmul6_cast_add_add(lv37: T.Buffer((T.int64(4096), T.int64(1376)), "uint32"), lv38: T.Buffer((T.int64(4096), T.int64(344)), "float16"), p_lv105: T.handle, p_lv41: T.handle, p_lv26: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv105 = T.match_buffer(p_lv105, (T.int64(1), n, T.int64(11008)), "float16")
    lv41 = T.match_buffer(p_lv41, (T.int64(1), n, T.int64(4096)), "float16")
    lv26 = T.match_buffer(p_lv26, (T.int64(1), n, T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)))
    var_compute_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv37[v_i, v_j // T.int64(8)], lv38[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv37[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv38[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(11008)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv105[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv105[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_i2, v_k])
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_compute_intermediate[v_ax0, v_ax1, v_ax2], lv41[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate[v_ax0, v_ax1, v_ax2] + lv41[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv26[v_ax0, v_ax1, v_ax2], var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv26[v_ax0, v_ax1, v_ax2] + var_T_add_intermediate[v_ax0, v_ax1, v_ax2]

# from tvm.script import tir as T

@T.prim_func
def fused_fused_decode3_fused_NT_matmul21_cast21_add5(lv2613: T.Buffer((T.int64(11008), T.int64(512)), "uint32"), lv2614: T.Buffer((T.int64(11008), T.int64(128)), "float16"), p_lv7341: T.handle, p_lv2616: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    b = T.int64()
    lv7341 = T.match_buffer(p_lv7341, (b, T.int64(512), T.int64(4096)), "float16")
    lv2616 = T.match_buffer(p_lv2616, (b, T.int64(512), T.int64(11008)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (b, T.int64(512), T.int64(11008)), "float16")
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(11008)))
    var_compute_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(11008)), "float16")
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv2613[v_i, v_j // T.int64(8)], lv2614[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv2613[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv2614[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(b, T.int64(512), T.int64(11008), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv7341[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv7341[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_i2, v_k])
    for i0, i1, i2 in T.grid(b, T.int64(512), T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(b, T.int64(512), T.int64(11008)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_compute_intermediate[v_ax0, v_ax1, v_ax2], lv2616[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate[v_ax0, v_ax1, v_ax2] + lv2616[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_fused_decode2_fused_NT_matmul_cast_add(lv4: T.Buffer((T.int64(4096), T.int64(512)), "uint32"), lv5: T.Buffer((T.int64(4096), T.int64(128)), "float16"), p_lv7: T.handle, p_lv7_1: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv7 = T.match_buffer(p_lv7, (T.int64(1), n, T.int64(4096)), "float16")
    lv7_1 = T.match_buffer(p_lv7_1, (T.int64(1), n, T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)))
    var_compute_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv4[v_i, v_j // T.int64(8)], lv5[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv4[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv5[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv7[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv7[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_i2, v_k])
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_compute_intermediate[v_ax0, v_ax1, v_ax2], lv7_1[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate[v_ax0, v_ax1, v_ax2] + lv7_1[v_ax0, v_ax1, v_ax2]

# from tvm.script import tir as T

@T.prim_func
def fused_fused_decode2_fused_NT_matmul_cast_add_add(lv22: T.Buffer((T.int64(4096), T.int64(512)), "uint32"), lv23: T.Buffer((T.int64(4096), T.int64(128)), "float16"), p_lv68: T.handle, p_lv25: T.handle, p_lv2: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv68 = T.match_buffer(p_lv68, (T.int64(1), n, T.int64(4096)), "float16")
    lv25 = T.match_buffer(p_lv25, (T.int64(1), n, T.int64(4096)), "float16")
    lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)))
    var_compute_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv22[v_i, v_j // T.int64(8)], lv23[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv22[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv23[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv68[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv68[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_i2, v_k])
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_compute_intermediate[v_ax0, v_ax1, v_ax2], lv25[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate[v_ax0, v_ax1, v_ax2] + lv25[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv2[v_ax0, v_ax1, v_ax2], var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv2[v_ax0, v_ax1, v_ax2] + var_T_add_intermediate[v_ax0, v_ax1, v_ax2]

# from tvm.script import tir as T

@T.prim_func
def fused_fused_decode1_fused_NT_matmul25_cast23_cast24(lv4019: T.Buffer((T.int64(32000), T.int64(512)), "uint32"), lv4020: T.Buffer((T.int64(32000), T.int64(128)), "float16"), p_lv10530: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    b = T.int64()
    lv10530 = T.match_buffer(p_lv10530, (b, T.int64(511), T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (b, T.int64(511), T.int64(32000)))
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(32000), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((b, T.int64(511), T.int64(32000)))
    var_compute_intermediate = T.alloc_buffer((b, T.int64(511), T.int64(32000)), "float16")
    for i, j in T.grid(T.int64(32000), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv4019[v_i, v_j // T.int64(8)], lv4020[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv4019[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv4020[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(b, T.int64(511), T.int64(32000), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv10530[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv10530[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_i2, v_k])
    for i0, i1, i2 in T.grid(b, T.int64(511), T.int64(32000)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
    for i0, i1, i2 in T.grid(b, T.int64(511), T.int64(32000)):
        with T.block("compute_1"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_compute_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_compute_intermediate[v_i0, v_i1, v_i2])


@T.prim_func
def fused_fused_decode4_fused_NT_matmul23_cast16_add4_add4(lv2618: T.Buffer((T.int64(4096), T.int64(1376)), "uint32"), lv2619: T.Buffer((T.int64(4096), T.int64(344)), "float16"), p_lv7353: T.handle, p_lv2622: T.handle, p_lv2607: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    b = T.int64()
    lv7353 = T.match_buffer(p_lv7353, (b, T.int64(512), T.int64(11008)), "float16")
    lv2622 = T.match_buffer(p_lv2622, (b, T.int64(512), T.int64(4096)), "float16")
    lv2607 = T.match_buffer(p_lv2607, (b, T.int64(512), T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (b, T.int64(512), T.int64(4096)), "float16")
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(4096)))
    var_compute_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(4096)), "float16")
    var_T_add_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv2618[v_i, v_j // T.int64(8)], lv2619[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv2618[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv2619[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(b, T.int64(512), T.int64(4096), T.int64(11008)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv7353[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv7353[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_i2, v_k])
    for i0, i1, i2 in T.grid(b, T.int64(512), T.int64(4096)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(b, T.int64(512), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_compute_intermediate[v_ax0, v_ax1, v_ax2], lv2622[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate[v_ax0, v_ax1, v_ax2] + lv2622[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(b, T.int64(512), T.int64(4096)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv2607[v_ax0, v_ax1, v_ax2], var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv2607[v_ax0, v_ax1, v_ax2] + var_T_add_intermediate[v_ax0, v_ax1, v_ax2]

# from tvm.script import tir as T

@T.prim_func
def fused_fused_decode4_transpose5_fused_NT_matmul21_cast21(lv4075: T.Buffer((T.int64(4096), T.int64(1376)), "uint32"), lv4076: T.Buffer((T.int64(4096), T.int64(344)), "float16"), p_lv10517_adjoint: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    b = T.int64()
    lv10517_adjoint = T.match_buffer(p_lv10517_adjoint, (b, T.int64(512), T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (b, T.int64(512), T.int64(11008)), "float16")
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(11008)))
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv4075[v_i, v_j // T.int64(8)], lv4076[v_i, v_j // T.int64(32)])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv4075[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv4076[v_i, v_j // T.int64(32)]
    for ax0, ax1 in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_decode_intermediate[v_ax1, v_ax0])
            T.writes(p_output0_intermediate_1[v_ax0, v_ax1])
            p_output0_intermediate_1[v_ax0, v_ax1] = var_decode_intermediate[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(b, T.int64(512), T.int64(11008), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv10517_adjoint[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv10517_adjoint[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_i2, v_k])
    for i0, i1, i2 in T.grid(b, T.int64(512), T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])

# from tvm.script import tir as T

@T.prim_func
def fused_fused_decode3_fused_NT_matmul4_cast5_add1_silu(lv27: T.Buffer((T.int64(11008), T.int64(512)), "uint32"), lv28: T.Buffer((T.int64(11008), T.int64(128)), "float16"), p_lv82: T.handle, p_lv30: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv82 = T.match_buffer(p_lv82, (T.int64(1), n, T.int64(4096)), "float16")
    lv30 = T.match_buffer(p_lv30, (T.int64(1), n, T.int64(11008)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(11008)), "float16")
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(11008)))
    var_compute_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(11008)), "float16")
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(11008)), "float16")
    compute = T.alloc_buffer((T.int64(1), n, T.int64(11008)), "float16")
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv27[v_i, v_j // T.int64(8)], lv28[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv27[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv28[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(11008), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv82[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv82[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_i2, v_k])
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_compute_intermediate[v_ax0, v_ax1, v_ax2], lv30[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate[v_ax0, v_ax1, v_ax2] + lv30[v_ax0, v_ax1, v_ax2]
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("compute_1"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_add_intermediate[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(var_T_add_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2], compute[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * compute[v_ax0, v_ax1, v_ax2]

# from tvm.script import tir as T

@T.prim_func
def fused_fused_decode3_fused_NT_matmul21_cast21_add5_silu2(lv2608: T.Buffer((T.int64(11008), T.int64(512)), "uint32"), lv2609: T.Buffer((T.int64(11008), T.int64(128)), "float16"), p_lv7330: T.handle, p_lv2611: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    b = T.int64()
    lv7330 = T.match_buffer(p_lv7330, (b, T.int64(512), T.int64(4096)), "float16")
    lv2611 = T.match_buffer(p_lv2611, (b, T.int64(512), T.int64(11008)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (b, T.int64(512), T.int64(11008)), "float16")
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(11008)))
    var_compute_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(11008)), "float16")
    var_T_add_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(11008)), "float16")
    compute = T.alloc_buffer((b, T.int64(512), T.int64(11008)), "float16")
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv2608[v_i, v_j // T.int64(8)], lv2609[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv2608[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv2609[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(b, T.int64(512), T.int64(11008), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv7330[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv7330[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_i2, v_k])
    for i0, i1, i2 in T.grid(b, T.int64(512), T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(b, T.int64(512), T.int64(11008)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_compute_intermediate[v_ax0, v_ax1, v_ax2], lv2611[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate[v_ax0, v_ax1, v_ax2] + lv2611[v_ax0, v_ax1, v_ax2]
    for i0, i1, i2 in T.grid(b, T.int64(512), T.int64(11008)):
        with T.block("compute_1"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_add_intermediate[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(var_T_add_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(b, T.int64(512), T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2], compute[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * compute[v_ax0, v_ax1, v_ax2]

# from tvm.script import tir as T

@T.prim_func
def fused_fused_decode2_fused_NT_matmul17_cast16_add4_add4(lv2603: T.Buffer((T.int64(4096), T.int64(512)), "uint32"), lv2604: T.Buffer((T.int64(4096), T.int64(128)), "float16"), p_lv7316: T.handle, p_lv2606: T.handle, p_lv7259: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    b = T.int64()
    lv7316 = T.match_buffer(p_lv7316, (b, T.int64(512), T.int64(4096)), "float16")
    lv2606 = T.match_buffer(p_lv2606, (b, T.int64(512), T.int64(4096)), "float16")
    lv7259 = T.match_buffer(p_lv7259, (b, T.int64(512), T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (b, T.int64(512), T.int64(4096)), "float16")
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(4096)))
    var_compute_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(4096)), "float16")
    var_T_add_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv2603[v_i, v_j // T.int64(8)], lv2604[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv2603[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv2604[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(b, T.int64(512), T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv7316[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv7316[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_i2, v_k])
    for i0, i1, i2 in T.grid(b, T.int64(512), T.int64(4096)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(b, T.int64(512), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_compute_intermediate[v_ax0, v_ax1, v_ax2], lv2606[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate[v_ax0, v_ax1, v_ax2] + lv2606[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(b, T.int64(512), T.int64(4096)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv7259[v_ax0, v_ax1, v_ax2], var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv7259[v_ax0, v_ax1, v_ax2] + var_T_add_intermediate[v_ax0, v_ax1, v_ax2]

# from tvm.script import tir as T


# from tvm.script import tir as T

@T.prim_func
def fused_fused_decode2_fused_NT_matmul17_cast16_add4(lv2579: T.Buffer((T.int64(4096), T.int64(512)), "uint32"), lv2580: T.Buffer((T.int64(4096), T.int64(128)), "float16"), p_lv7265: T.handle, p_lv2582: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    b = T.int64()
    lv7265 = T.match_buffer(p_lv7265, (b, T.int64(512), T.int64(4096)), "float16")
    lv2582 = T.match_buffer(p_lv2582, (b, T.int64(512), T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (b, T.int64(512), T.int64(4096)), "float16")
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(4096)))
    var_compute_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv2579[v_i, v_j // T.int64(8)], lv2580[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv2579[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv2580[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(b, T.int64(512), T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv7265[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv7265[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_i2, v_k])
    for i0, i1, i2 in T.grid(b, T.int64(512), T.int64(4096)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(b, T.int64(512), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_compute_intermediate[v_ax0, v_ax1, v_ax2], lv2582[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate[v_ax0, v_ax1, v_ax2] + lv2582[v_ax0, v_ax1, v_ax2]

# from tvm.script import tir as T

@T.prim_func
def fused_fused_decode3_transpose_fused_NT_matmul23_cast16(lv4082: T.Buffer((T.int64(11008), T.int64(512)), "uint32"), lv4083: T.Buffer((T.int64(11008), T.int64(128)), "float16"), p_lv10505_adjoint: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    b = T.int64()
    lv10505_adjoint = T.match_buffer(p_lv10505_adjoint, (b, T.int64(512), T.int64(11008)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (b, T.int64(512), T.int64(4096)), "float16")
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(4096)))
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv4082[v_i, v_j // T.int64(8)], lv4083[v_i, v_j // T.int64(32)])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv4082[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv4083[v_i, v_j // T.int64(32)]
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_decode_intermediate[v_ax1, v_ax0])
            T.writes(p_output0_intermediate_1[v_ax0, v_ax1])
            p_output0_intermediate_1[v_ax0, v_ax1] = var_decode_intermediate[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(b, T.int64(512), T.int64(4096), T.int64(11008)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv10505_adjoint[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv10505_adjoint[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_i2, v_k])
    for i0, i1, i2 in T.grid(b, T.int64(512), T.int64(4096)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])

@T.prim_func
def fused_fused_decode2_transpose13_fused_NT_matmul17_cast16(lv4097: T.Buffer((T.int64(4096), T.int64(512)), "uint32"), lv4098: T.Buffer((T.int64(4096), T.int64(128)), "float16"), p_lv10480_adjoint: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    b = T.int64()
    lv10480_adjoint = T.match_buffer(p_lv10480_adjoint, (b, T.int64(512), T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (b, T.int64(512), T.int64(4096)), "float16")
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(4096)))
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv4097[v_i, v_j // T.int64(8)], lv4098[v_i, v_j // T.int64(32)])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv4097[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv4098[v_i, v_j // T.int64(32)]
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_decode_intermediate[v_ax1, v_ax0])
            T.writes(p_output0_intermediate_1[v_ax0, v_ax1])
            p_output0_intermediate_1[v_ax0, v_ax1] = var_decode_intermediate[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(b, T.int64(512), T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv10480_adjoint[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv10480_adjoint[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_i2, v_k])
    for i0, i1, i2 in T.grid(b, T.int64(512), T.int64(4096)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])

# from tvm.script import tir as T

@T.prim_func
def fused_fused_decode3_fused_NT_matmul4_cast5_add1(lv32: T.Buffer((T.int64(11008), T.int64(512)), "uint32"), lv33: T.Buffer((T.int64(11008), T.int64(128)), "float16"), p_lv93: T.handle, p_lv35: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv93 = T.match_buffer(p_lv93, (T.int64(1), n, T.int64(4096)), "float16")
    lv35 = T.match_buffer(p_lv35, (T.int64(1), n, T.int64(11008)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(11008)), "float16")
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(11008)))
    var_compute_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(11008)), "float16")
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv32[v_i, v_j // T.int64(8)], lv33[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv32[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv33[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(11008), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv93[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv93[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_i2, v_k])
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_compute_intermediate[v_ax0, v_ax1, v_ax2], lv35[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate[v_ax0, v_ax1, v_ax2] + lv35[v_ax0, v_ax1, v_ax2]

# from tvm.script import tir as T

@T.prim_func
def fused_fused_decode1_transpose2_fused_NT_matmul26_cast25(lv4024: T.Buffer((T.int64(32000), T.int64(512)), "uint32"), lv4025: T.Buffer((T.int64(32000), T.int64(128)), "float16"), p_lv4023: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    b = T.int64()
    lv4023 = T.match_buffer(p_lv4023, (b, T.int64(511), T.int64(32000)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (b, T.int64(511), T.int64(4096)), "float16")
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(32000), T.int64(4096)), "float16")
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(32000)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((b, T.int64(511), T.int64(4096)))
    for i, j in T.grid(T.int64(32000), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv4024[v_i, v_j // T.int64(8)], lv4025[v_i, v_j // T.int64(32)])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv4024[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv4025[v_i, v_j // T.int64(32)]
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(32000)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_decode_intermediate[v_ax1, v_ax0])
            T.writes(p_output0_intermediate_1[v_ax0, v_ax1])
            p_output0_intermediate_1[v_ax0, v_ax1] = var_decode_intermediate[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(b, T.int64(511), T.int64(4096), T.int64(32000)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv4023[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv4023[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_i2, v_k])
    for i0, i1, i2 in T.grid(b, T.int64(511), T.int64(4096)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])

@T.prim_func(private=True)
def main(p_lv10476_adjoint: T.handle, p_lv10467_cp: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    b = T.int64()
    lv10476_adjoint = T.match_buffer(p_lv10476_adjoint, (b, T.int64(32), T.int64(512), T.int64(128)), "float16")
    lv10467_cp = T.match_buffer(p_lv10467_cp, (b, T.int64(32), T.int64(512), T.int64(128)), "float16")
    var_compute_intermediate = T.match_buffer(p_output0, (b, T.int64(32), T.int64(512), T.int64(512)))
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((b, T.int64(32), T.int64(512), T.int64(512)))
    var_compute_intermediate_1 = T.alloc_buffer((b, T.int64(32), T.int64(512), T.int64(512)), "float16")
    for i0, i1, i2, i3, k in T.grid(b, T.int64(32), T.int64(512), T.int64(512), T.int64(128)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(lv10476_adjoint[v_i0, v_i1, v_i2, v_k], lv10467_cp[v_i0, v_i1, v_i3, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + T.Cast("float32", lv10476_adjoint[v_i0, v_i1, v_i2, v_k]) * T.Cast("float32", lv10467_cp[v_i0, v_i1, v_i3, v_k])
    for i0, i1, i2, i3 in T.grid(b, T.int64(32), T.int64(512), T.int64(512)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(var_compute_intermediate_1[v_i0, v_i1, v_i2, v_i3])
            var_compute_intermediate_1[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
    for i0, i1, i2, i3 in T.grid(b, T.int64(32), T.int64(512), T.int64(512)):
        with T.block("compute_1"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_compute_intermediate_1[v_i0, v_i1, v_i2, v_i3])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
            var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float32", var_compute_intermediate_1[v_i0, v_i1, v_i2, v_i3])

@T.prim_func(private=True)
def fused_matmul1_cast13(p_lv7313: T.handle, p_lv7305: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    b = T.int64()
    lv7313 = T.match_buffer(p_lv7313, (b, T.int64(32), T.int64(512), T.int64(512)), "float16")
    lv7305 = T.match_buffer(p_lv7305, (b, T.int64(32), T.int64(512), T.int64(128)), "float16")
    var_compute_intermediate = T.match_buffer(p_output0, (b, T.int64(32), T.int64(512), T.int64(128)), "float16")
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((b, T.int64(32), T.int64(512), T.int64(128)))
    for i0, i1, i2, i3, k in T.grid(b, T.int64(32), T.int64(512), T.int64(128), T.int64(512)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(lv7313[v_i0, v_i1, v_i2, v_k], lv7305[v_i0, v_i1, v_k, v_i3])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + T.Cast("float32", lv7313[v_i0, v_i1, v_i2, v_k]) * T.Cast("float32", lv7305[v_i0, v_i1, v_k, v_i3])
    for i0, i1, i2, i3 in T.grid(b, T.int64(32), T.int64(512), T.int64(128)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
            var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])

@T.prim_func(private=True)
def fused_TN_matmul2_cast13(p_lv10475_cp: T.handle, p_lv10476_adjoint: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    b = T.int64()
    lv10475_cp = T.match_buffer(p_lv10475_cp, (b, T.int64(32), T.int64(512), T.int64(512)), "float16")
    lv10476_adjoint = T.match_buffer(p_lv10476_adjoint, (b, T.int64(32), T.int64(512), T.int64(128)), "float16")
    var_compute_intermediate = T.match_buffer(p_output0, (b, T.int64(32), T.int64(512), T.int64(128)), "float16")
    # with T.block("root"):
    var_T_matmul_TN_intermediate = T.alloc_buffer((b, T.int64(32), T.int64(512), T.int64(128)))
    for i0, i1, i2, i3, k in T.grid(b, T.int64(32), T.int64(512), T.int64(128), T.int64(512)):
        with T.block("T_matmul_TN"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(lv10475_cp[v_i0, v_i1, v_k, v_i2], lv10476_adjoint[v_i0, v_i1, v_k, v_i3])
            T.writes(var_T_matmul_TN_intermediate[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                var_T_matmul_TN_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            var_T_matmul_TN_intermediate[v_i0, v_i1, v_i2, v_i3] = var_T_matmul_TN_intermediate[v_i0, v_i1, v_i2, v_i3] + T.Cast("float32", lv10475_cp[v_i0, v_i1, v_k, v_i2]) * T.Cast("float32", lv10476_adjoint[v_i0, v_i1, v_k, v_i3])
    for i0, i1, i2, i3 in T.grid(b, T.int64(32), T.int64(512), T.int64(128)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_matmul_TN_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
            var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", var_T_matmul_TN_intermediate[v_i0, v_i1, v_i2, v_i3])
