import math
import time
import torch

from config import *
from flashattn.tir_impl import attn_forward_profile
from tir_impl import attn_backward
from torch_impl import Attention
from utils import timeit, do_bench


torch.manual_seed(1234)


step_1 = True
step_2 = False
step_3 = False
step_4 = True
enable_number_output = True
enable_fallback_dtype = True
enable_fallback_dtype = enable_fallback_dtype and dtype_str != fallback_dtype_str
enable_check_result = True


scale = 1 / math.sqrt(head_dim)


shape_q = (batch, n_head, seq, head_dim)
shape_k = (batch, n_head, seq, head_dim)
shape_v = (batch, n_head, seq, head_dim)

# inputs
q = torch.randn(shape_q, requires_grad=True, dtype=dtype_torch, device=dev_torch)
k = torch.randn(shape_k, requires_grad=True, dtype=dtype_torch, device=dev_torch)
v = torch.randn(shape_v, requires_grad=True, dtype=dtype_torch, device=dev_torch)


# 1. use pytorch built-in operator
if step_1:
    print("-------------------torch builtin-------------------")
    import torch.backends.cuda

    # with torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=False):
    timeit("Torch Builtin Forward")
    if enable_fallback_dtype:
        output = torch.nn.functional.scaled_dot_product_attention(
            q.to(fallback_dtype_torch), k.to(fallback_dtype_torch), v.to(fallback_dtype_torch)
        ).to(dtype_torch)
    else:
        # slight difference
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    timeit("Torch Builtin Forward")

    if enable_number_output:
        print("Forward output:", output[0, 0])

    dO = torch.ones_like(output, device=dev_torch)
    timeit("Torch Builtin Backward")
    output.backward(dO)
    timeit("Torch Builtin Backward")

    if enable_number_output:
        print("Gradient w.r.t q:", q.grad[0, 0])
        print("Gradient w.r.t k:", k.grad[0, 0])
        print("Gradient w.r.t v:", v.grad[0, 0])

    if enable_check_result:
        output_std = output.clone().detach().cpu()
        q_grad_std = q.grad.clone().detach().cpu()
        k_grad_std = k.grad.clone().detach().cpu()
        v_grad_std = v.grad.clone().detach().cpu()

    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()


# 2. use attention implemented by torch.autograd.Function
if step_2:
    print("-------------------torch autograd.Function impl-------------------")
    op = Attention.apply

    timeit("Torch Autograd Forward")
    output = op(q, k, v, scale)
    timeit("Torch Autograd Forward")

    if enable_number_output:
        print("Forward output:", output[0, 0])

    if enable_check_result:
        assert torch.allclose(output_std, output.detach().cpu(), rtol=rtol, atol=atol)

    dO = torch.ones_like(output, device=dev_torch)
    timeit("Torch Autograd Backward")
    output.backward(dO)
    timeit("Torch Autograd Backward")

    if enable_number_output:
        print("Gradient w.r.t q:", q.grad[0, 0])
        print("Gradient w.r.t k:", k.grad[0, 0])
        print("Gradient w.r.t v:", v.grad[0, 0])

    if enable_check_result:
        assert torch.allclose(q_grad_std, q.grad.detach().cpu(), rtol=rtol, atol=atol)
        assert torch.allclose(k_grad_std, k.grad.detach().cpu(), rtol=rtol, atol=atol)
        assert torch.allclose(v_grad_std, v.grad.detach().cpu(), rtol=rtol, atol=atol)

    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()


# 3. use attention implemented by combination of torch functional operators
if step_3:
    print("-------------------torch operator impl-------------------")

    timeit("Torch Op Forward")
    if enable_fallback_dtype:
        output = torch.nn.functional.softmax(((q @ k.transpose(-2, -1)) * scale), dim=-1) @ v
    else:
        # no difference
        output = torch.nn.functional.softmax(((q @ k.transpose(-2, -1)) * scale).to(fallback_dtype_torch), dim=-1).to(dtype_torch) @ v
    timeit("Torch Op Forward")

    if enable_number_output:
        print("Forward output:", output[0, 0])

    if enable_check_result:
        assert torch.allclose(output_std, output.detach().cpu(), rtol=rtol, atol=atol)

    dO = torch.ones_like(output, device=dev_torch)
    timeit("Torch Op Backward")
    output.backward(dO)
    timeit("Torch Op Backward")

    if enable_number_output:
        print("Gradient w.r.t q:", q.grad[0, 0])
        print("Gradient w.r.t k:", k.grad[0, 0])
        print("Gradient w.r.t v:", v.grad[0, 0])

    if enable_check_result:
        assert torch.allclose(q_grad_std, q.grad.detach().cpu(), rtol=rtol, atol=atol)
        assert torch.allclose(k_grad_std, k.grad.detach().cpu(), rtol=rtol, atol=atol)
        assert torch.allclose(v_grad_std, v.grad.detach().cpu(), rtol=rtol, atol=atol)

    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()


# 4. use Attention implemented by TVM
if step_4:
    print("-------------------TVM impl-------------------")
    from tir_impl import attn_forward

    for i in range(10):
        timeit("TVM Forward", 1, 0)
        output, L, *rest = attn_forward(q, k, v)
        timeit("TVM Forward", 1, 0)

    if enable_number_output:
        print("Forward output:", output[0, 0])
        if len(rest):
            print("Forward debug:", *rest, sep="\n")

    if enable_check_result:
        assert torch.allclose(output_std, output.detach().cpu(), rtol=rtol, atol=atol)

    attn_forward_profile(q, k, v)

    # dO = torch.ones_like(output, device=dev_torch)
    # timeit("Backward")
    # dQ, dK, dV, *rest = attn_backward(dO, q, k, v, output, L)
    # timeit("Backward")

    # if enable_number_output:
    #     if len(rest):
    #         print("Backward debug:", *rest, sep="\n")
    #     print("Gradient w.r.t q:", dQ[0, 0])
    #     print("Gradient w.r.t k:", dK[0, 0])
    #     print("Gradient w.r.t v:", dV[0, 0])

    # if enable_check_result:
    #     assert torch.allclose(q_grad_std, dQ.detach().cpu(), rtol=rtol, atol=atol)
    #     assert torch.allclose(k_grad_std, dK.detach().cpu(), rtol=rtol, atol=atol)
    #     assert torch.allclose(v_grad_std, dV.detach().cpu(), rtol=rtol, atol=atol)
