from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import tvm
from tvm import relax, te, tir, topi
from tvm.script import tir as T
from tvm.relax.expr_functor import visitor


@T.prim_func
def decode1(A: T.Buffer((T.int64(32000), T.int64(512)), "uint32"), B: T.Buffer((T.int64(32000), T.int64(128)), "float16"), decode: T.Buffer((T.int64(32000), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i, j in T.grid(T.int64(32000), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(A[v_i, v_j // T.int64(8)], B[v_i, v_j // T.int64(32)])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * B[v_i, v_j // T.int64(32)]

@T.prim_func
def decode2(A: T.Buffer((T.int64(4096), T.int64(512)), "uint32"), B: T.Buffer((T.int64(4096), T.int64(128)), "float16"), decode: T.Buffer((T.int64(4096), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(A[v_i, v_j // T.int64(8)], B[v_i, v_j // T.int64(32)])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * B[v_i, v_j // T.int64(32)]

@T.prim_func
def decode3(A: T.Buffer((T.int64(11008), T.int64(512)), "uint32"), B: T.Buffer((T.int64(11008), T.int64(128)), "float16"), decode: T.Buffer((T.int64(11008), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(A[v_i, v_j // T.int64(8)], B[v_i, v_j // T.int64(32)])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * B[v_i, v_j // T.int64(32)]

@T.prim_func
def decode4(A: T.Buffer((T.int64(4096), T.int64(1376)), "uint32"), B: T.Buffer((T.int64(4096), T.int64(344)), "float16"), decode: T.Buffer((T.int64(4096), T.int64(11008)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(A[v_i, v_j // T.int64(8)], B[v_i, v_j // T.int64(32)])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * B[v_i, v_j // T.int64(32)]
