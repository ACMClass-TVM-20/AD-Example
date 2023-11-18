from typing import List, Optional
import time

import numpy as np

import tvm
from tvm import dlight as dl, relax, tir
from tvm.ir import IRModule
from tvm.ir.transform import PassContext, module_pass
from tvm.target import Target
from tvm.script import ir as I, relax as R, tir as T

from config import *

def try_inline(sch: tir.Schedule):
    root = sch.get_block("root")
    blocks = sch.get_child_blocks(root)
    flag = True
    while flag:
        flag = False
        for block in blocks:
            try:
                sch.compute_inline(block)
                flag = True
            except:
                continue
        for block in blocks:
            try:
                sch.reverse_compute_inline(block)
                flag = True
            except:
                continue


def schedule_forward_op(func: tir.PrimFunc) -> tir.PrimFunc:
    sch = tir.Schedule(func, enable_check=False)
    outer_loops = sch.fuse(*sch.get_loops("outer_compute"))
    sch.bind(outer_loops, "blockIdx.x")

    inner_factor = Br
    outer_factor = 128 // inner_factor

    # load_Qi
    v0, v1 = sch.get_loops("load_Qi")
    _, outer, inner = sch.split(v1, factors=[None, outer_factor, inner_factor])
    sch.bind(outer, "threadIdx.y")
    sch.bind(inner, "threadIdx.x")

    # init_Oi
    v0, v1 = sch.get_loops("init_Oi")
    _, outer, inner = sch.split(v1, factors=[None, outer_factor, inner_factor])
    sch.bind(outer, "threadIdx.y")
    sch.bind(inner, "threadIdx.x")

    # load_Kj
    v0, v1 = sch.get_loops("load_Kj")
    _, outer, inner = sch.split(v1, factors=[None, outer_factor, inner_factor])
    sch.bind(outer, "threadIdx.y")
    sch.bind(inner, "threadIdx.x")

    # compute_Sij
    v0, v1, v2 = sch.get_loops("compute_Sij")
    _, outer, inner = sch.split(v2, factors=[None, outer_factor, inner_factor])
    sch.bind(outer, "threadIdx.y")
    sch.bind(inner, "threadIdx.x")

    # store_mi_old
    # v0 = sch.get_loops("store_mi_old")
    # _, outer, inner = sch.split(v1, factors=[None, outer_factor, inner_factor])
    # sch.bind(outer, "threadIdx.y")

    # compute_mi_mid
    # v0, v1 = sch.get_loops("compute_mi_mid")
    # mid, inner = sch.split(v1, factors=[None, inner_factor])
    # _, outer = sch.split(v0, factors=[None, outer_factor])
    # sch.bind(outer, "threadIdx.y")
    # sch.bind(inner, "threadIdx.x")

    # compute_Pij
    v0, v1 = sch.get_loops("compute_Pij")
    _, outer, inner = sch.split(sch.fuse(v0, v1), factors=[None, outer_factor, inner_factor])
    sch.bind(outer, "threadIdx.y")
    sch.bind(inner, "threadIdx.x")

    # compute_li
    # v0, v1 = sch.get_loops("compute_li")
    # _, outer = sch.split(v0, factors=[None, outer_factor])
    # sch.bind(outer, "threadIdx.y")
    # sch.bind(v1, "threadIdx.x")

    # load_Vj
    v0, v1 = sch.get_loops("load_Vj")
    _, outer, inner = sch.split(v1, factors=[None, outer_factor, inner_factor])
    sch.bind(outer, "threadIdx.y")
    sch.bind(inner, "threadIdx.x")

    # compute_Oi_mid
    v0, v1, v2 = sch.get_loops("compute_Oi")
    _, outer, inner = sch.split(v1, factors=[None, outer_factor, inner_factor])
    sch.bind(outer, "threadIdx.y")
    sch.bind(inner, "threadIdx.x")

    # compute_Oi
    # v0, v1 = sch.get_loops("compute_Oi")
    # _, outer, inner = sch.split(v1, factors=[None, outer_factor, inner_factor])
    # sch.bind(outer, "threadIdx.y")
    # sch.bind(inner, "threadIdx.x")

    # store_Oi
    v0, v1 = sch.get_loops("store_Oi")
    _, outer, inner = sch.split(v1, factors=[None, outer_factor, inner_factor])
    sch.bind(outer, "threadIdx.y")
    sch.bind(inner, "threadIdx.x")

    # store_L
    (v0,) = sch.get_loops("store_L")
    _, outer, inner = sch.split(v0, factors=[None, outer_factor, inner_factor])
    sch.bind(outer, "threadIdx.y")
    sch.bind(inner, "threadIdx.x")

    return sch.mod["main"]


def schedule_backward_op(func: tir.PrimFunc) -> tir.PrimFunc:
    sch = tir.Schedule(func, enable_check=False)
    compute_D_loops = sch.get_loops("compute_D")
    sch.bind(sch.fuse(*compute_D_loops[:2]), "blockIdx.x")
    sch.bind(compute_D_loops[3], "threadIdx.x")

    outer_compute_loops = sch.get_loops("outer_compute")
    sch.bind(outer_compute_loops[0], "blockIdx.z")
    sch.bind(outer_compute_loops[1], "blockIdx.y")
    sch.bind(outer_compute_loops[2], "blockIdx.x")

    thread_count = 128
    inner_factor = Br
    outer_factor = thread_count // inner_factor

    v0, v1 = sch.get_loops("load_Kj")
    sch.bind(v1, "threadIdx.x")

    v0, v1 = sch.get_loops("load_Vj")
    sch.bind(v1, "threadIdx.x")

    v0, v1 = sch.get_loops("init_dKj")
    sch.bind(v1, "threadIdx.x")

    v0, v1 = sch.get_loops("init_dVj")
    sch.bind(v1, "threadIdx.x")

    v0, v1 = sch.get_loops("load_Qi")
    sch.bind(v1, "threadIdx.x")

    v0, v1 = sch.get_loops("load_dOi")
    sch.bind(v1, "threadIdx.x")

    v0, v1, v2 = sch.get_loops("compute_Sij")
    fused = sch.fuse(v0, v1)
    sch.bind(sch.split(fused, [None, thread_count])[1], "threadIdx.x")
    # sch.bind(v2, "threadIdx.x")

    v0, v1 = sch.get_loops("compute_Pij")
    fused = sch.fuse(v0, v1)
    sch.bind(sch.split(fused, [None, thread_count])[1], "threadIdx.x")

    v0, v1, v2 = sch.get_loops("compute_dVj")
    sch.bind(v2, "threadIdx.x")

    v0, v1, v2 = sch.get_loops("compute_dPij")
    fused = sch.fuse(v0, v1)
    sch.bind(sch.split(fused, [None, thread_count])[1], "threadIdx.x")
    # sch.bind(v2, "threadIdx.x")

    v0, v1 = sch.get_loops("compute_dSij")
    fused = sch.fuse(v0, v1)
    sch.bind(sch.split(fused, [None, thread_count])[1], "threadIdx.x")

    v0, v1, v2 = sch.get_loops("compute_dQi")
    sch.bind(v1, "threadIdx.x")

    v0, v1 = sch.get_loops("store_dQi_cache")
    sch.bind(v1, "threadIdx.x")

    v0, v1, v2 = sch.get_loops("compute_dKj")
    sch.bind(v2, "threadIdx.x")

    v0, v1 = sch.get_loops("store_dKj")
    sch.bind(v1, "threadIdx.x")

    v0, v1 = sch.get_loops("store_dVj")
    sch.bind(v1, "threadIdx.x")

    b, nh, v0, v1 = sch.get_loops("store_dQ")
    fused = sch.fuse(b, nh, v0, v1)
    outer, inner = sch.split(fused, [None, thread_count])
    sch.bind(outer, "blockIdx.x")
    sch.bind(inner, "threadIdx.x")

    return sch.mod["main"]
