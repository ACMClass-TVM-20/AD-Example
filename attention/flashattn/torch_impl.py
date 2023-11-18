import torch

from config import *


def load_tile_from_global(tensor, batch, block_size, idx):
    return tensor[batch, block_size * idx : block_size * (idx + 1)]


def save_tile_to_global(tensor, batch, block_size, idx, new_val):
    tensor[batch, block_size * idx : block_size * (idx + 1)] = new_val


# torch impl of Flash Attention 2
class Attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        scale: float = 1.0,
    ):
        to_save = [Q, K, V]
        ctx.scale = scale
        orig_shape = Q.shape
        dtype = Q.dtype
        Q = Q.view(-1, Q.shape[-2], Q.shape[-1])
        K = K.view(-1, K.shape[-2], K.shape[-1])
        V = V.view(-1, V.shape[-2], V.shape[-1])

        B, N, d = Q.shape
        Tr = N // Br
        Tc = N // Bc
        O = torch.zeros((B, N, d), dtype=dtype, device=dev_torch)
        L = torch.zeros((B, N), dtype=dtype, device=dev_torch)

        for b in range(B):
            for i in range(Tr):
                Qi = load_tile_from_global(Q, b, Br, i)
                Oi = torch.zeros((Br, d), dtype=dtype, device=dev_torch)
                li = torch.zeros((Br,), dtype=dtype, device=dev_torch)
                mi = torch.full((Br,), torch.finfo(dtype).min, dtype=dtype, device=dev_torch)
                for j in range(Tc):
                    Kj = load_tile_from_global(K, b, Bc, j)
                    Vj = load_tile_from_global(V, b, Bc, j)
                    Sij = scale * (Qi @ Kj.T)
                    mi_old = mi
                    mi = torch.maximum(mi, torch.max(Sij, dim=-1).values)
                    Pij = torch.exp(Sij - mi.reshape(-1, 1))
                    li = torch.exp(mi_old - mi) * li + torch.sum(Pij, dim=-1)
                    Oi = Oi * torch.exp(mi_old - mi).reshape(-1, 1) + Pij @ Vj

                Oi = Oi / li.reshape(-1, 1)
                save_tile_to_global(O, b, Br, i, Oi)
                Li = mi + torch.log(li)
                save_tile_to_global(L, b, Br, i, Li)

        O = O.view(*orig_shape)

        to_save += [O, L]
        ctx.save_for_backward(*to_save)
        return O

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, dO: torch.Tensor):
        Q, K, V, O, L = ctx.saved_tensors
        scale = ctx.scale

        orig_shape = Q.shape
        Q = Q.view(-1, Q.shape[-2], Q.shape[-1])
        K = K.view(-1, K.shape[-2], K.shape[-1])
        V = V.view(-1, V.shape[-2], V.shape[-1])
        O = O.view(-1, O.shape[-2], O.shape[-1])
        dO = dO.view(-1, dO.shape[-2], dO.shape[-1])

        B, N, d = Q.shape
        Tr = N // Br
        Tc = N // Bc

        dQ = torch.zeros_like(Q, device=dev_torch)
        dK = torch.zeros_like(K, device=dev_torch)
        dV = torch.zeros_like(V, device=dev_torch)
        D = torch.sum(dO * O, dim=-1)

        for b in range(B):
            for j in range(Tc):
                Kj = load_tile_from_global(K, b, Bc, j)
                Vj = load_tile_from_global(V, b, Bc, j)
                dKj = torch.zeros_like(Kj, device=dev_torch)
                dVj = torch.zeros_like(Vj, device=dev_torch)
                for i in range(Tr):
                    Qi = load_tile_from_global(Q, b, Br, i)
                    dOi = load_tile_from_global(dO, b, Br, i)
                    Li = load_tile_from_global(L, b, Br, i)
                    Di = load_tile_from_global(D, b, Br, i)
                    Sij = scale * (Qi @ Kj.T)
                    Pij = torch.exp(Sij - Li.reshape(-1, 1))
                    dVj = dVj + Pij.T @ dOi
                    dPij = dOi @ Vj.T
                    dSij = Pij * (dPij - Di.reshape(-1, 1))
                    dQi = load_tile_from_global(dQ, b, Br, i)
                    dQi = dQi + scale * (dSij @ Kj)
                    save_tile_to_global(dQ, b, Br, i, dQi)
                    dKj = dKj + scale * (dSij.T @ Qi)
                save_tile_to_global(dK, b, Bc, j, dKj)
                save_tile_to_global(dV, b, Bc, j, dVj)

        dQ = dQ.view(*orig_shape)
        dK = dK.view(*orig_shape)
        dV = dV.view(*orig_shape)
        return dQ, dK, dV, None
