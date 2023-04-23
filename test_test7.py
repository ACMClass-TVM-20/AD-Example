import torch


import torch
import torch.utils.checkpoint
import torch.nn as nn


# def foo(x, y):
#     out = torch.sum(2 * x + y)
#     out.backward()
#     return x.grad


# # Run `foo` with the provided inputs and record the tensor operations
# inputs = [
#     torch.rand(3, dtype=torch.float32, requires_grad=True),
#     torch.rand(3, dtype=torch.float32, requires_grad=True),
# ]
# # print(foo(*inputs))
# traced_foo = torch.jit.trace(foo, inputs)
# print(traced_foo.code)

torch.utils.checkpoint.checkpoint
