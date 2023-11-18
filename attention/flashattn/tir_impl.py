import math
import os
import tvm
from tvm import relax, tir, te, topi
from tvm.ir.module import IRModule
from tvm.relax.analysis import estimate_memory_usage
from tvm.relax.block_builder import BlockBuilder
from tvm.relay import GlobalVar
from tvm.target.target import Target
import tvm.testing
from tvm.script.parser import relax as R, tir as T, ir as I
import pytest
import numpy as np
from tvm.relax.expr_functor import mutator, PyExprMutator
from tvm.relax.transform.tuning_api import Trace
from tvm.tir.schedule.schedule import Schedule

from tvm.relax.dpl.pattern import is_op, wildcard
import torch
import tvm.dlight as dl

import sch_rules


from config import *


shape_q = (batch, n_head, seq, head_dim)
shape_k = (batch, n_head, seq, head_dim)
shape_v = (batch, n_head, seq, head_dim)
shape_o = (batch, n_head, seq, head_dim)
shape_l = (batch, n_head, seq)
scale = 1 / math.sqrt(head_dim)
scale_tir = tir.const(scale, dtype=dtype_str)


@I.ir_module
class Module:
    @T.prim_func
    def forward_op(
        Q: T.Buffer(shape_q, dtype_str),
        K: T.Buffer(shape_k, dtype_str),
        V: T.Buffer(shape_v, dtype_str),
        O: T.Buffer(shape_o, dtype_str),
        L: T.Buffer(shape_l, dtype_str),
    ):
        Qi = T.alloc_buffer((Br, head_dim), dtype=dtype_str, scope="shared")
        Kj = T.alloc_buffer((Bc, head_dim), dtype=dtype_str, scope="shared")
        # Vj = T.alloc_buffer((Bc, head_dim), dtype=dtype_str, scope="shared")
        Oi = T.alloc_buffer((Br, head_dim), dtype=fallback_dtype_str, scope="shared")
        # Oi_mid = T.alloc_buffer((Br, head_dim), dtype=fallback_dtype_str, scope="shared")
        li = T.alloc_buffer((Br,), dtype=fallback_dtype_str, scope="local")
        mi = T.alloc_buffer((Br,), dtype=fallback_dtype_str, scope="local")
        mi_mid = T.alloc_buffer((Br,), dtype=fallback_dtype_str, scope="local")
        mi_old = T.alloc_buffer((Br,), dtype=fallback_dtype_str, scope="local")
        Sij = T.alloc_buffer((Br, Bc), dtype=fallback_dtype_str, scope="shared")
        Pij = T.alloc_buffer((Br, Bc), dtype=fallback_dtype_str, scope="shared")

        for b, nh in T.grid(batch, n_head):
            for i in range(Tr):
                with T.block("outer_compute"):
                    ab, anh, ai = T.axis.remap("SSS", [b, nh, i])
                    # Qi = load_tile_from_global(Q, b, Attention.Br, i)
                    for v0, v1 in T.grid(Br, head_dim):
                        with T.block("load_Qi"):
                            av0, av1 = T.axis.remap("SS", [v0, v1])
                            Qi[av0, av1] = Q[ab, anh, ai * Br + av0, av1]
                    # Oi = torch.zeros((Br, head_dim), dtype=dtype)
                    for v0, v1 in T.grid(Br, head_dim):
                        with T.block("init_Oi"):
                            av0, av1 = T.axis.remap("SS", [v0, v1])
                            Oi[av0, av1] = tir.const(0.0, dtype_str)
                    # li = torch.zeros((Br,), dtype=dtype)
                    for v0 in range(Br):
                        with T.block("init_li"):
                            av0 = T.axis.S(Br, v0)
                            li[av0] = tir.const(0.0, fallback_dtype_str)
                    # mi = torch.zeros((Br,), dtype=dtype)
                    for v0 in range(Br):
                        with T.block("init_mi"):
                            av0 = T.axis.S(Br, v0)
                            mi[av0] = tvm.tir.min_value(fallback_dtype_str)
                    for j in range(Tc):
                        with T.block("inner_compute"):
                            aj = T.axis.R(Tc, j)
                            # Kj = load_tile_from_global(K, b, Attention.Bc, j)
                            for v0, v1 in T.grid(Bc, head_dim):
                                with T.block("load_Kj"):
                                    av0, av1 = T.axis.remap("SS", [v0, v1])
                                    Kj[av0, av1] = K[ab, anh, aj * Bc + av0, av1]
                            # Sij = scale * (Qi @ Kj^T)
                            for v0, v1, v2 in T.grid(Br, Bc, head_dim):
                                with T.block("compute_Sij"):
                                    av0, av1, av2 = T.axis.remap("SSR", [v0, v1, v2])
                                    with T.init():
                                        Sij[av0, av1] = tir.const(0.0, fallback_dtype_str)
                                    Sij[av0, av1] = Sij[av0, av1] + tir.Cast(fallback_dtype_str, Qi[av0, av2] * Kj[av1, av2] * scale_tir)
                            # mi_old = mi
                            for v0 in range(Br):
                                with T.block("store_mi_old"):
                                    av0 = T.axis.remap("S", [v0])
                                    mi_old[av0] = mi[av0]
                            # mi = torch.maximum(mi, torch.max(Sij, dim=-1).values)
                            for v0, v1 in T.grid(Br, Bc):
                                with T.block("compute_mi_mid"):
                                    av0, av1 = T.axis.remap("SR", [v0, v1])
                                    with T.init():
                                        mi_mid[av0] = tvm.tir.min_value(fallback_dtype_str)
                                    mi_mid[av0] = T.max(mi_mid[av0], Sij[av0, av1])
                            for v0 in range(Br):
                                with T.block("compute_mi"):
                                    av0 = T.axis.remap("S", [v0])
                                    mi[av0] = T.max(mi_old[av0], mi_mid[av0])
                            # Pij = torch.exp(Sij - mi.reshape(-1, 1))
                            for v0, v1 in T.grid(Br, Bc):
                                with T.block("compute_Pij"):
                                    av0, av1 = T.axis.remap("SS", [v0, v1])
                                    Pij[av0, av1] = T.exp(Sij[av0, av1] - mi[av0])
                            # li = torch.exp(mi_old - mi) * li + torch.sum(Pij, dim=-1)
                            for v0, v1 in T.grid(Br, Bc):
                                with T.block("compute_li"):
                                    av0, av1 = T.axis.remap("SR", [v0, v1])
                                    with T.init():
                                        li[av0] = T.exp(mi_old[av0] - mi[av0]) * li[av0]
                                    li[av0] = li[av0] + Pij[av0, av1]
                            # Vj = load_tile_from_global(V, b, Attention.Bc, j)
                            # here Kj is an alias of Vj
                            for v0, v1 in T.grid(Bc, head_dim):
                                with T.block("load_Vj"):
                                    av0, av1 = T.axis.remap("SS", [v0, v1])
                                    Kj[av0, av1] = V[ab, anh, aj * Bc + av0, av1]
                            # Oi = Oi * torch.exp(mi_old - mi).reshape(-1, 1) + Pij @ Vj
                            for v0, v1, v2 in T.grid(Br, head_dim, Bc):
                                with T.block("compute_Oi"):
                                    av0, av1, av2 = T.axis.remap("SSR", [v0, v1, v2])
                                    with T.init():
                                        Oi[av0, av1] = T.exp(mi_old[av0] - mi[av0]) * Oi[av0, av1]
                                    Oi[av0, av1] = Oi[av0, av1] + tir.Cast(fallback_dtype_str, Pij[av0, av2] * Kj[av2, av1])
                            # for v0, v1, v2 in T.grid(Br, head_dim, Bc):
                            #     with T.block("compute_Oi_mid"):
                            #         av0, av1, av2 = T.axis.remap("SSR", [v0, v1, v2])
                            #         with T.init():
                            #             Oi_mid[av0, av1] = tir.const(0.0, fallback_dtype_str)
                            #         Oi_mid[av0, av1] = Oi_mid[av0, av1] + tir.Cast(fallback_dtype_str, Pij[av0, av2] * Kj[av2, av1])
                            # for v0, v1 in T.grid(Br, head_dim):
                            #     with T.block("compute_Oi"):
                            #         av0, av1 = T.axis.remap("SS", [v0, v1])
                            #         Oi[av0, av1] = Oi_mid[av0, av1] + T.exp(mi_old[av0] - mi[av0]) * Oi[av0, av1]
                    # Oi = Oi / li.reshape(-1, 1)
                    # save_tile_to_global(O, b, Attention.Br, i, Oi)
                    for v0, v1 in T.grid(Br, head_dim):
                        with T.block("store_Oi"):
                            av0, av1 = T.axis.remap("SS", [v0, v1])
                            O[ab, anh, ai * Br + av0, av1] = tir.Cast(dtype_str, Oi[av0, av1] / li[av0])
                    # Li = mi + torch.log(li)
                    # save_tile_to_global(L, b, Attention.Br, i, Li)
                    for v0 in range(Br):
                        with T.block("store_L"):
                            av0 = T.axis.remap("S", [v0])
                            L[ab, anh, ai * Br + av0] = tir.Cast(dtype_str, mi[av0] + T.log(li[av0]))

    # @T.prim_func
    # def backward_op(
    #     dO: T.Buffer(shape_o, dtype_str),
    #     Q: T.Buffer(shape_q, dtype_str),
    #     K: T.Buffer(shape_k, dtype_str),
    #     V: T.Buffer(shape_v, dtype_str),
    #     O: T.Buffer(shape_o, dtype_str),
    #     L: T.Buffer(shape_l, dtype_str),
    #     dQ: T.Buffer(shape_q, dtype_str),
    #     dK: T.Buffer(shape_k, dtype_str),
    #     dV: T.Buffer(shape_v, dtype_str),
    # ):
    #     Qi = T.alloc_buffer((Br, head_dim), dtype=dtype_str, scope="shared")
    #     Kj = T.alloc_buffer((Bc, head_dim), dtype=dtype_str, scope="shared")
    #     Vj = T.alloc_buffer((Bc, head_dim), dtype=dtype_str, scope="shared")
    #     Li = T.alloc_buffer((Br,), dtype=dtype_str, scope="shared")
    #     Di = T.alloc_buffer((Br,), dtype=dtype_str, scope="shared")
    #     dOi = T.alloc_buffer((Br, head_dim), dtype=dtype_str, scope="shared")
    #     dQi = T.alloc_buffer((Br, head_dim), dtype=dtype_str, scope="shared")
    #     dKj = T.alloc_buffer((Bc, head_dim), dtype=dtype_str, scope="shared")
    #     dVj = T.alloc_buffer((Bc, head_dim), dtype=dtype_str, scope="shared")
    #     Sij = T.alloc_buffer((Br, Bc), dtype=dtype_str, scope="shared")
    #     Pij = T.alloc_buffer((Br, Bc), dtype=dtype_str, scope="shared")
    #     dSij = T.alloc_buffer((Br, Bc), dtype=dtype_str, scope="shared")
    #     dPij = T.alloc_buffer((Br, Bc), dtype=dtype_str, scope="shared")
    #     D = T.alloc_buffer((batch, n_head, seq), dtype=dtype_str)
    #     # D = torch.sum(dO * O, dim=-1)
    #     for b, nh, seq, head_dim in T.grid(batch, n_head, seq, head_dim):
    #         with T.block("compute_D"):
    #             ab, anh, aseq, ahead_dim = T.axis.remap("SSSR", [b, nh, seq, head_dim])
    #             with T.init():
    #                 D[ab, anh, aseq] = tir.const(0.0, dtype_str)
    #             D[ab, anh, aseq] = D[ab, anh, aseq] + dO[ab, anh, aseq, ahead_dim] * O[ab, anh, aseq, ahead_dim]
    #     # dQ = torch.zeros_like(Q)
    #     for b, nh, seq, head_dim in T.grid(batch, n_head, seq, head_dim):
    #         with T.block("init_dQ"):
    #             ab, anh, aseq, ahead_dim = T.axis.remap("SSSS", [b, nh, seq, head_dim])
    #             dQ[ab, anh, aseq, ahead_dim] = tir.const(0.0, dtype_str)
    #     for b, nh in T.grid(batch, n_head):
    #         for j in range(Tc):
    #             with T.block("outer_compute"):
    #                 ab, anh, aj = T.axis.remap("SSR", [b, nh, j])
    #                 # Kj = load_tile_from_global(K, b, Attention.Bc, j)
    #                 for v0, v1 in T.grid(Bc, head_dim):
    #                     with T.block("load_Kj"):
    #                         av0, av1 = T.axis.remap("SS", [v0, v1])
    #                         Kj[av0, av1] = K[ab, anh, aj * Bc + av0, av1]
    #                 # Vj = load_tile_from_global(V, b, Attention.Bc, j)
    #                 for v0, v1 in T.grid(Bc, head_dim):
    #                     with T.block("load_Vj"):
    #                         av0, av1 = T.axis.remap("SS", [v0, v1])
    #                         Vj[av0, av1] = V[ab, anh, aj * Bc + av0, av1]
    #                 # dKj = torch.zeros_like(Kj)
    #                 for v0, v1 in T.grid(Bc, head_dim):
    #                     with T.block("init_dKj"):
    #                         av0, av1 = T.axis.remap("SS", [v0, v1])
    #                         dKj[av0, av1] = tir.const(0.0, dtype_str)
    #                 # dVj = torch.zeros_like(Vj)
    #                 for v0, v1 in T.grid(Bc, head_dim):
    #                     with T.block("init_dVj"):
    #                         av0, av1 = T.axis.remap("SS", [v0, v1])
    #                         dVj[av0, av1] = tir.const(0.0, dtype_str)
    #                 for i in range(Tr):
    #                     with T.block("inner_compute"):
    #                         ai = T.axis.R(Tr, i)
    #                         # Qi = load_tile_from_global(Q, b, Attention.Br, i)
    #                         for v0, v1 in T.grid(Br, head_dim):
    #                             with T.block("load_Qi"):
    #                                 av0, av1 = T.axis.remap("SS", [v0, v1])
    #                                 Qi[av0, av1] = Q[ab, anh, ai * Br + av0, av1]
    #                         # dOi = load_tile_from_global(dO, b, Attention.Br, i)
    #                         for v0, v1 in T.grid(Br, head_dim):
    #                             with T.block("load_dOi"):
    #                                 av0, av1 = T.axis.remap("SS", [v0, v1])
    #                                 dOi[av0, av1] = dO[ab, anh, ai * Br + av0, av1]
    #                         # Li = load_tile_from_global(L, b, Attention.Br, i)
    #                         for v0 in range(Br):
    #                             with T.block("load_Li"):
    #                                 av0 = T.axis.S(Br, v0)
    #                                 Li[av0] = L[ab, anh, ai * Br + av0]
    #                         # Di = load_tile_from_global(D, b, Attention.Br, i)
    #                         for v0 in range(Br):
    #                             with T.block("load_Di"):
    #                                 av0 = T.axis.S(Br, v0)
    #                                 Di[av0] = D[ab, anh, ai * Br + av0]
    #                         # Sij = scale * (Qi @ Kj^T)
    #                         for v0, v1, v2 in T.grid(Br, Bc, head_dim):
    #                             with T.block("compute_Sij"):
    #                                 av0, av1, av2 = T.axis.remap("SSR", [v0, v1, v2])
    #                                 with T.init():
    #                                     Sij[av0, av1] = tir.const(0.0, dtype_str)
    #                                 Sij[av0, av1] = Sij[av0, av1] + Qi[av0, av2] * Kj[av1, av2] * scale_tir
    #                         # Pij = torch.exp(Sij - Li.reshape(-1, 1))
    #                         for v0, v1 in T.grid(Br, Bc):
    #                             with T.block("compute_Pij"):
    #                                 av0, av1 = T.axis.remap("SS", [v0, v1])
    #                                 Pij[av0, av1] = T.exp(Sij[av0, av1] - Li[av0])
    #                         # dVj = dVj + Pij.T @ dOi
    #                         for v0, v1, v2 in T.grid(Bc, Br, head_dim):
    #                             with T.block("compute_dVj"):
    #                                 av0, av1, av2 = T.axis.remap("SRS", [v0, v1, v2])
    #                                 dVj[av0, av2] = dVj[av0, av2] + Pij[av1, av0] * dOi[av1, av2]
    #                         # dPij = dOi @ Vj^T
    #                         for v0, v1, v2 in T.grid(Br, Bc, head_dim):
    #                             with T.block("compute_dPij"):
    #                                 av0, av1, av2 = T.axis.remap("SSR", [v0, v1, v2])
    #                                 with T.init():
    #                                     dPij[av0, av1] = tir.const(0.0, dtype_str)
    #                                 dPij[av0, av1] = dPij[av0, av1] + dOi[av0, av2] * Vj[av1, av2]
    #                         # dSij = Pij * (dPij - Di.reshape(-1, 1))
    #                         for v0, v1 in T.grid(Br, Bc):
    #                             with T.block("compute_dSij"):
    #                                 av0, av1 = T.axis.remap("SS", [v0, v1])
    #                                 dSij[av0, av1] = Pij[av0, av1] * (dPij[av0, av1] - Di[av0])
    #                         # dQi = load_tile_from_global(dQ, b, Attention.Br, i)
    #                         for v0, v1 in T.grid(Br, head_dim):
    #                             with T.block("load_dQi"):
    #                                 av0, av1 = T.axis.remap("SS", [v0, v1])
    #                                 dQi[av0, av1] = dQ[ab, anh, ai * Br + av0, av1]
    #                         # dQi = dQi + scale * (dSij @ Kj)
    #                         for v0, v1, v2 in T.grid(Br, Bc, head_dim):
    #                             with T.block("compute_dQi"):
    #                                 av0, av1, av2 = T.axis.remap("SRS", [v0, v1, v2])
    #                                 dQi[av0, av2] = dQi[av0, av2] + scale_tir * dSij[av0, av1] * Kj[av1, av2]
    #                         # save_tile_to_global(dQ, b, Attention.Br, i, dQi)
    #                         for v0, v1 in T.grid(Br, head_dim):
    #                             with T.block("store_dQi"):
    #                                 av0, av1 = T.axis.remap("SS", [v0, v1])
    #                                 T.at
    #                                 dQ[ab, anh, ai * Br + av0, av1] = dQi[av0, av1]
    #                         # dKj = dKj + scale * (dSij.T @ Qi)
    #                         for v0, v1, v2 in T.grid(Bc, Br, head_dim):
    #                             with T.block("compute_dKj"):
    #                                 av0, av1, av2 = T.axis.remap("SRS", [v0, v1, v2])
    #                                 dKj[av0, av2] = dKj[av0, av2] + scale_tir * dSij[av1, av0] * Qi[av1, av2]
    #                 # save_tile_to_global(dK, b, Attention.Bc, j, dKj)
    #                 for v0, v1 in T.grid(Bc, head_dim):
    #                     with T.block("store_dKj"):
    #                         av0, av1 = T.axis.remap("SS", [v0, v1])
    #                         dK[ab, anh, aj * Bc + av0, av1] = dKj[av0, av1]
    #                 # save_tile_to_global(dV, b, Attention.Bc, j, dVj)
    #                 for v0, v1 in T.grid(Bc, head_dim):
    #                     with T.block("store_dVj"):
    #                         av0, av1 = T.axis.remap("SS", [v0, v1])
    #                         dV[ab, anh, aj * Bc + av0, av1] = dVj[av0, av1]

    @T.prim_func
    def backward_op(
        dO: T.Buffer(shape_o, dtype_str),
        Q: T.Buffer(shape_q, dtype_str),
        K: T.Buffer(shape_k, dtype_str),
        V: T.Buffer(shape_v, dtype_str),
        O: T.Buffer(shape_o, dtype_str),
        L: T.Buffer(shape_l, dtype_str),
        dQ: T.Buffer(shape_q, dtype_str),
        dK: T.Buffer(shape_k, dtype_str),
        dV: T.Buffer(shape_v, dtype_str),
    ):
        Qi = T.alloc_buffer((Br, head_dim), dtype=dtype_str, scope="shared")
        Kj = T.alloc_buffer((Bc, head_dim), dtype=dtype_str, scope="shared")
        Vj = T.alloc_buffer((Bc, head_dim), dtype=dtype_str, scope="shared")
        Li = T.alloc_buffer((Br,), dtype=dtype_str, scope="local")
        Di = T.alloc_buffer((Br,), dtype=dtype_str, scope="local")
        dOi = T.alloc_buffer((Br, head_dim), dtype=dtype_str, scope="shared")
        dQi = T.alloc_buffer((Br, head_dim), dtype=fallback_dtype_str, scope="shared")
        dKj = T.alloc_buffer((Bc, head_dim), dtype=fallback_dtype_str, scope="shared")
        dVj = T.alloc_buffer((Bc, head_dim), dtype=fallback_dtype_str, scope="shared")
        Sij = T.alloc_buffer((Br, Bc), dtype=fallback_dtype_str, scope="shared")
        Pij = T.alloc_buffer((Br, Bc), dtype=fallback_dtype_str, scope="shared")
        dSij = T.alloc_buffer((Br, Bc), dtype=fallback_dtype_str, scope="shared")
        dPij = T.alloc_buffer((Br, Bc), dtype=fallback_dtype_str, scope="shared")
        D = T.alloc_buffer((batch, n_head, seq), dtype=dtype_str)
        dQ_cache = T.alloc_buffer((batch, n_head, Tc, seq, head_dim), dtype=dtype_str)
        # D = torch.sum(dO * O, dim=-1)
        for b, nh, seq, head_dim in T.grid(batch, n_head, seq, head_dim):
            with T.block("compute_D"):
                ab, anh, aseq, ahead_dim = T.axis.remap("SSSR", [b, nh, seq, head_dim])
                with T.init():
                    D[ab, anh, aseq] = tir.const(0.0, dtype_str)
                D[ab, anh, aseq] = D[ab, anh, aseq] + dO[ab, anh, aseq, ahead_dim] * O[ab, anh, aseq, ahead_dim]
        for b, nh in T.grid(batch, n_head):
            for j in range(Tc):
                with T.block("outer_compute"):
                    ab, anh, aj = T.axis.remap("SSS", [b, nh, j])
                    # Kj = load_tile_from_global(K, b, Attention.Bc, j)
                    for v0, v1 in T.grid(Bc, head_dim):
                        with T.block("load_Kj"):
                            av0, av1 = T.axis.remap("SS", [v0, v1])
                            Kj[av0, av1] = K[ab, anh, aj * Bc + av0, av1]
                    # Vj = load_tile_from_global(V, b, Attention.Bc, j)
                    for v0, v1 in T.grid(Bc, head_dim):
                        with T.block("load_Vj"):
                            av0, av1 = T.axis.remap("SS", [v0, v1])
                            Vj[av0, av1] = V[ab, anh, aj * Bc + av0, av1]
                    # dKj = torch.zeros_like(Kj)
                    for v0, v1 in T.grid(Bc, head_dim):
                        with T.block("init_dKj"):
                            av0, av1 = T.axis.remap("SS", [v0, v1])
                            dKj[av0, av1] = tir.const(0.0, dtype_str)
                    # dVj = torch.zeros_like(Vj)
                    for v0, v1 in T.grid(Bc, head_dim):
                        with T.block("init_dVj"):
                            av0, av1 = T.axis.remap("SS", [v0, v1])
                            dVj[av0, av1] = tir.const(0.0, fallback_dtype_str)
                    for i in range(Tr):
                        with T.block("inner_compute"):
                            ai = T.axis.R(Tr, i)
                            # Qi = load_tile_from_global(Q, b, Attention.Br, i)
                            for v0, v1 in T.grid(Br, head_dim):
                                with T.block("load_Qi"):
                                    av0, av1 = T.axis.remap("SS", [v0, v1])
                                    Qi[av0, av1] = Q[ab, anh, ai * Br + av0, av1]
                            # dOi = load_tile_from_global(dO, b, Attention.Br, i)
                            for v0, v1 in T.grid(Br, head_dim):
                                with T.block("load_dOi"):
                                    av0, av1 = T.axis.remap("SS", [v0, v1])
                                    dOi[av0, av1] = dO[ab, anh, ai * Br + av0, av1]
                            # Li = load_tile_from_global(L, b, Attention.Br, i)
                            for v0 in range(Br):
                                with T.block("load_Li"):
                                    av0 = T.axis.S(Br, v0)
                                    Li[av0] = L[ab, anh, ai * Br + av0]
                            # Di = load_tile_from_global(D, b, Attention.Br, i)
                            for v0 in range(Br):
                                with T.block("load_Di"):
                                    av0 = T.axis.S(Br, v0)
                                    Di[av0] = D[ab, anh, ai * Br + av0]
                            # Sij = scale * (Qi @ Kj^T)
                            for v0, v1, v2 in T.grid(Br, Bc, head_dim):
                                with T.block("compute_Sij"):
                                    av0, av1, av2 = T.axis.remap("SSR", [v0, v1, v2])
                                    with T.init():
                                        Sij[av0, av1] = tir.const(0.0, fallback_dtype_str)
                                    Sij[av0, av1] = tir.Cast(fallback_dtype_str, Sij[av0, av1] + Qi[av0, av2] * Kj[av1, av2] * scale_tir)
                            # Pij = torch.exp(Sij - Li.reshape(-1, 1))
                            for v0, v1 in T.grid(Br, Bc):
                                with T.block("compute_Pij"):
                                    av0, av1 = T.axis.remap("SS", [v0, v1])
                                    Pij[av0, av1] = T.exp(Sij[av0, av1] - tir.Cast(fallback_dtype_str, Li[av0]))
                            # dVj = dVj + Pij.T @ dOi
                            for v0, v1, v2 in T.grid(Bc, Br, head_dim):
                                with T.block("compute_dVj"):
                                    av0, av1, av2 = T.axis.remap("SRS", [v0, v1, v2])
                                    dVj[av0, av2] = dVj[av0, av2] + Pij[av1, av0] * tir.Cast(fallback_dtype_str, dOi[av1, av2])
                            # dPij = dOi @ Vj^T
                            for v0, v1, v2 in T.grid(Br, Bc, head_dim):
                                with T.block("compute_dPij"):
                                    av0, av1, av2 = T.axis.remap("SSR", [v0, v1, v2])
                                    with T.init():
                                        dPij[av0, av1] = tir.const(0.0, dtype_str)
                                    dPij[av0, av1] = dPij[av0, av1] + tir.Cast(fallback_dtype_str, dOi[av0, av2] * Vj[av1, av2])
                            # dSij = Pij * (dPij - Di.reshape(-1, 1))
                            for v0, v1 in T.grid(Br, Bc):
                                with T.block("compute_dSij"):
                                    av0, av1 = T.axis.remap("SS", [v0, v1])
                                    dSij[av0, av1] = Pij[av0, av1] * (dPij[av0, av1] - tir.Cast(fallback_dtype_str, Di[av0]))
                            # dQi = scale * (dSij @ Kj)
                            for v0, v1, v2 in T.grid(Br, head_dim, Bc):
                                with T.block("compute_dQi"):
                                    av0, av1, av2 = T.axis.remap("SSR", [v0, v1, v2])
                                    with T.init():
                                        dQi[av0, av1] = tir.const(0.0, fallback_dtype_str)
                                    dQi[av0, av1] = dQi[av0, av1] + tir.Cast(fallback_dtype_str, scale_tir) * dSij[av0, av2] * (
                                        tir.Cast(fallback_dtype_str, Kj[av2, av1])
                                    )
                            # save_tile_to_global(dQ_cache, b, Attention.Br, i, dQi)
                            for v0, v1 in T.grid(Br, head_dim):
                                with T.block("store_dQi_cache"):
                                    av0, av1 = T.axis.remap("SS", [v0, v1])
                                    dQ_cache[ab, anh, aj, ai * Br + av0, av1] = tir.Cast(dtype_str, dQi[av0, av1])
                            # dKj = dKj + scale * (dSij.T @ Qi)
                            for v0, v1, v2 in T.grid(Bc, Br, head_dim):
                                with T.block("compute_dKj"):
                                    av0, av1, av2 = T.axis.remap("SRS", [v0, v1, v2])
                                    dKj[av0, av2] = dKj[av0, av2] + tir.Cast(fallback_dtype_str, scale_tir) * dSij[av1, av0] * (
                                        tir.Cast(fallback_dtype_str, Qi[av1, av2])
                                    )
                    # save_tile_to_global(dK, b, Attention.Bc, j, dKj)
                    for v0, v1 in T.grid(Bc, head_dim):
                        with T.block("store_dKj"):
                            av0, av1 = T.axis.remap("SS", [v0, v1])
                            dK[ab, anh, aj * Bc + av0, av1] = tir.Cast(dtype_str, dKj[av0, av1])
                    # save_tile_to_global(dV, b, Attention.Bc, j, dVj)
                    for v0, v1 in T.grid(Bc, head_dim):
                        with T.block("store_dVj"):
                            av0, av1 = T.axis.remap("SS", [v0, v1])
                            dV[ab, anh, aj * Bc + av0, av1] = tir.Cast(dtype_str, dVj[av0, av1])
        accumulator = T.alloc_buffer((1,), dtype=fallback_dtype_str, scope="local")
        for b, nh in T.grid(batch, n_head):
            # combine dQ_cache to dQ
            for v0, v1 in T.grid(seq, head_dim):
                with T.block("store_dQ"):
                    ab, anh, av0, av1 = T.axis.remap("SSSS", [b, nh, v0, v1])
                    for j in range(Tc):
                        with T.block("combine_dQ"):
                            aj = T.axis.remap("R", [j])
                            with T.init():
                                accumulator[0] = tir.const(0.0, fallback_dtype_str)
                            accumulator[0] = accumulator[0] + tir.Cast(fallback_dtype_str, dQ_cache[ab, anh, aj, av0, av1])
                    dQ[ab, anh, av0, av1] = tir.Cast(dtype_str, accumulator[0])

    @R.function
    def forward(Q: R.Tensor(shape_q, dtype_str), K: R.Tensor(shape_k, dtype_str), V: R.Tensor(shape_v, dtype_str)):
        cls = Module
        out = R.call_tir(cls.forward_op, (Q, K, V), out_sinfo=[R.Tensor(shape_o, dtype_str), R.Tensor(shape_l, dtype_str)])
        return out

    @R.function
    def backward(
        dO: R.Tensor(shape_o, dtype_str),
        Q: R.Tensor(shape_q, dtype_str),
        K: R.Tensor(shape_k, dtype_str),
        V: R.Tensor(shape_v, dtype_str),
        O: R.Tensor(shape_o, dtype_str),
        L: R.Tensor(shape_l, dtype_str),
    ):
        cls = Module
        out = R.call_tir(cls.backward_op, (dO, Q, K, V, O, L), out_sinfo=[R.Tensor(shape_q, dtype_str), R.Tensor(shape_k, dtype_str), R.Tensor(shape_v, dtype_str)])
        return out


mod = Module


if target_tvm != "llvm":
    with target_tvm:
        with open(os.path.join(path_prefix, "mod_before.py"), "w") as f_before:
            print(mod.script(), file=f_before, flush=True)
        mod["forward_op"] = sch_rules.schedule_forward_op(mod["forward_op"])
        mod["backward_op"] = sch_rules.schedule_backward_op(mod["backward_op"])

        print("Schedule done")
        with open(os.path.join(path_prefix, "mod_after.py"), "w") as f_after:
            print(mod.script(), file=f_after, flush=True)

ex = relax.build(mod, target=target_tvm)
vm = relax.VirtualMachine(ex, dev_tvm, profile=True)

if target_tvm != "llvm":
    with open(os.path.join(path_prefix, "mod_build.cu"), "w") as f_build:
        print(ex.mod.imported_modules[0].get_source(), file=f_build, flush=True)
print("build done")



def attn_forward_profile(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    inputs = [tvm.nd.array(x.detach().cpu().numpy(), dev_tvm) for x in [Q, K, V]]
    report = vm.profile("forward", *inputs)
    print(report)




def attn_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    inputs = [tvm.nd.array(x.detach().cpu().numpy(), dev_tvm) for x in [Q, K, V]]
    res = vm["forward"](*inputs)
    return [torch.from_numpy(x.numpy()) for x in res]


def attn_backward(dO: torch.Tensor, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor, L: torch.Tensor):
    inputs = [tvm.nd.array(x.detach().cpu().numpy(), dev_tvm) for x in [dO, Q, K, V, O, L]]
    res = vm["backward"](*inputs)
    return [torch.from_numpy(x.numpy()) for x in res]
