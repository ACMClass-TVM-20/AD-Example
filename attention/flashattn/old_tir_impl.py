import math
import tvm
from tvm import relax, tir, te, topi
from tvm.script import relax as R, tir as T, ir as I

batch = 6
n_head = 32
seq = 512
head_dim = 128
Bc = 16
Br = 16

Tr = seq // Br
Tc = seq // Bc

dtype_str = "float32"
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
    def forward_op_original(
        Q: T.Buffer(shape_q, dtype_str),
        K: T.Buffer(shape_k, dtype_str),
        V: T.Buffer(shape_v, dtype_str),
        O: T.Buffer(shape_o, dtype_str),
        L: T.Buffer(shape_l, dtype_str),
    ):
        Qi = T.alloc_buffer((Br, head_dim), dtype=dtype_str, scope="shared")
        Kj = T.alloc_buffer((Bc, head_dim), dtype=dtype_str, scope="shared")
        Vj = T.alloc_buffer((Bc, head_dim), dtype=dtype_str, scope="shared")
        Oi = T.alloc_buffer((Br, head_dim), dtype=dtype_str, scope="shared")
        li = T.alloc_buffer((Br,), dtype=dtype_str, scope="shared")
        mi = T.alloc_buffer((Br,), dtype=dtype_str, scope="shared")
        mi_old = T.alloc_buffer((Br,), dtype=dtype_str, scope="shared")
        Sij = T.alloc_buffer((Br, Bc), dtype=dtype_str, scope="shared")
        Pij = T.alloc_buffer((Br, Bc), dtype=dtype_str, scope="shared")

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
                            li[av0] = tir.const(0.0, dtype_str)
                    # mi = torch.zeros((Br,), dtype=dtype)
                    for v0 in range(Br):
                        with T.block("init_mi"):
                            av0 = T.axis.S(Br, v0)
                            mi[av0] = tvm.tir.min_value(dtype_str)
                    for j in range(Tc):
                        with T.block("inner_compute"):
                            aj = T.axis.R(Tc, j)
                            # Kj = load_tile_from_global(K, b, Attention.Bc, j)
                            # Vj = load_tile_from_global(V, b, Attention.Bc, j)
                            for v0, v1 in T.grid(Bc, head_dim):
                                with T.block("load_Kj"):
                                    av0, av1 = T.axis.remap("SS", [v0, v1])
                                    Kj[av0, av1] = K[ab, anh, aj * Bc + av0, av1]
                                with T.block("load_Vj"):
                                    av0, av1 = T.axis.remap("SS", [v0, v1])
                                    Vj[av0, av1] = V[ab, anh, aj * Bc + av0, av1]
                            # Sij = scale * (Qi @ Kj^T)
                            for v0, v1, v2 in T.grid(Br, Bc, head_dim):
                                with T.block("compute_Sij"):
                                    av0, av1, av2 = T.axis.remap("SSR", [v0, v1, v2])
                                    with T.init():
                                        Sij[av0, av1] = tir.const(0.0, dtype_str)
                                    Sij[av0, av1] = Sij[av0, av1] + Qi[av0, av2] * Kj[av1, av2] * scale_tir
                            # mi_old = mi
                            for v0 in range(Br):
                                with T.block("store_mi_old"):
                                    av0 = T.axis.remap("S", [v0])
                                    mi_old[av0] = mi[av0]
                            # mi = torch.maximum(mi, torch.max(Sij, dim=-1).values)
                            for v0, v1 in T.grid(Br, Bc):
                                with T.block("compute_mi"):
                                    av0, av1 = T.axis.remap("SR", [v0, v1])
                                    mi[av0] = T.max(mi[av0], Sij[av0, av1])
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
                            # Oi = Oi * torch.exp(mi_old - mi).reshape(-1, 1) + Pij @ Vj
                            for v0, v1, v2 in T.grid(Br, Bc, head_dim):
                                with T.block("compute_Oi"):
                                    av0, av1, av2 = T.axis.remap("SRS", [v0, v1, v2])
                                    with T.init():
                                        Oi[av0, av2] = T.exp(mi_old[av0] - mi[av0]) * Oi[av0, av2]
                                    Oi[av0, av2] = Oi[av0, av2] + Pij[av0, av1] * Vj[av1, av2]
                    # Oi = Oi / li.reshape(-1, 1)
                    # save_tile_to_global(O, b, Attention.Br, i, Oi)
                    for v0, v1 in T.grid(Br, head_dim):
                        with T.block("store_Oi"):
                            av0, av1 = T.axis.remap("SS", [v0, v1])
                            O[ab, anh, ai * Br + av0, av1] = Oi[av0, av1] / li[av0]
                    # Li = mi + torch.log(li)
                    # save_tile_to_global(L, b, Attention.Br, i, Li)
                    for v0 in range(Br):
                        with T.block("store_L"):
                            av0 = T.axis.remap("S", [v0])
                            L[ab, anh, ai * Br + av0] = mi[av0] + T.log(li[av0])
