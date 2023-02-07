"""This file shows how to use the Relax Optimizer APIs to optimize parameters.

We will use the SGD algorithm to minimize the L2 distance between input parameter x and label y.
Both of them are float32 Tensors of shape (3, 3).
"""

import numpy as np
import tvm
from tvm.relax.block_builder import BlockBuilder
from tvm.runtime.container import tuple_object
import tvm.script
from tvm import relax
from tvm.script.parser import ir as I, relax as R, tir as T
from tvm.relax.transform import LegalizeOps

import torch
import torchvision

train_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

img, label = next(iter(test_loader))
img = img.reshape(1, 28, 28).numpy()

# Define the input Vars and the forward function "main"
x = relax.Var("x", R.Tensor((3, 3), "float32"))
y = relax.Var("y", R.Tensor((3, 3), "float32"))

builder = BlockBuilder()
with builder.function("main", [x, y]):
    with builder.dataflow():
        lv = builder.emit(R.subtract(x, y))
        lv1 = builder.emit(R.multiply(lv, lv))
        gv = builder.emit_output(R.sum(lv1))
    builder.emit_func_output(gv)
mod = builder.get()

# AD process, differentiate "main" and generate a new function "main_adjoint"
mod = relax.transform.Gradient(mod.get_global_var("main"), x)(mod)

# Show the complete IRModule
mod.show()

# Optimizer function generation
# Note that `opt.state` would be used later to get the state of the optimizer
opt = relax.optimizer.SGD(x, 0.1).set_vm_config("llvm", tvm.cpu())

# Build and legalize module
lowered_mod = LegalizeOps()(mod)
ex = relax.vm.build(lowered_mod, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

# Runtime inputs
x_input = tvm.nd.array(np.random.rand(3, 3).astype(np.float32))
x_input_tuple = tuple_object([x_input])
y_input = tvm.nd.array(np.zeros((3, 3), "float32"))

# Training process
steps = 100
for i in range(steps):
    res, x_grad = vm["main_adjoint"](*x_input_tuple, y_input)
    x_input_tuple = opt(x_input_tuple, x_grad)
    print("Step:", i)
    print("loss =", res.numpy())
    print("x =", x_input_tuple[0].numpy(), "\n")
