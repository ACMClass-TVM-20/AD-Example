"""This file shows how to use the Relax Optimizer APIs to optimize parameters.

We will use the SGD algorithm to minimize the L2 distance between input parameter x and label y.
Both of them are float32 Tensors of shape (3, 3).
"""

import numpy as np
import tvm
from tvm.relax.block_builder import BlockBuilder
from tvm.relax.expr import Var
from tvm.relax.training.loss import CrossEntropyLoss
from tvm.relax.training.setup_trainer import SetupTrainer
from tvm.relax.training.trainer import Trainer
from tvm.runtime.container import tuple_object
import tvm.script
from tvm import relax
from tvm.script.parser import ir as I, relax as R, tir as T
from tvm.relax.transform import LegalizeOps
from tvm.relax.training.optimizer import Adam, SGD
from tvm.relax.transform import LegalizeOps, Gradient
from tvm.relax.training.loss import Loss
from tvm.relax.training.utils import append_loss
from tvm.relax.training.optimizer import Optimizer
from tvm.relax.struct_info import TensorStructInfo
from tvm.relax.analysis import well_formed


batch_size = 64

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as Tr
import torch.nn.functional as Func

train_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=Tr.Compose([Tr.ToTensor(), Tr.Lambda(torch.flatten)]),
    target_transform=lambda x:Func.one_hot(torch.tensor(x), 10).float()
)
test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=Tr.Compose([Tr.ToTensor(), Tr.Lambda(torch.flatten)]),
    target_transform=lambda x:Func.one_hot(torch.tensor(x), 10).float()
)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



@tvm.script.ir_module
class Backbone:
    @R.function
    def predict(
        w0: R.Tensor((784, 128), "float32"),
        b0: R.Tensor((128,), "float32"),
        w1: R.Tensor((128, 128), "float32"),
        b1: R.Tensor((128,), "float32"),
        w2: R.Tensor((128, 10), "float32"),
        b2: R.Tensor((10,), "float32"),
        x: R.Tensor((batch_size, 784), "float32"),
    ):
        with R.dataflow():
            lv0 = R.matmul(x, w0)
            lv1 = R.add(lv0, b0)
            lv2 = R.nn.relu(lv1)
            lv3 = R.matmul(lv2, w1)
            lv4 = R.add(lv3, b1)
            lv5 = R.nn.relu(lv4)
            lv6 = R.matmul(lv5, w2)
            out = R.add(lv6, b2)
            R.output(out)
        return out

out_sinfo = relax.TensorStructInfo((batch_size, 10), "float32")
label_sinfo = relax.TensorStructInfo((batch_size, 10), "float32")



func = CrossEntropyLoss(reduction="sum")(out_sinfo, label_sinfo)
print(func)
Backbone["loss"] = append_loss(Backbone["predict"], func)
Backbone.show()

params = Backbone["loss"].params[:6]

Backbone = relax.transform.Gradient(
    Backbone.get_global_var("loss"),
    require_grads=params
)(Backbone)
Backbone.show()


opt = relax.optimizer.SGD(0.1).init(params)
Backbone["SGD"] = opt.get_function()
print(Backbone["SGD"])

# Build and legalize module
lowered_mod = LegalizeOps()(Backbone)
ex = relax.vm.build(lowered_mod, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())


# TODO(chaofan, yixin): Support symbolic shapes
def _get_shape_as_int_list(var: Var):
    return [int(val) for val in var.struct_info.shape]


params_list = [tvm.nd.array(np.ones(_get_shape_as_int_list(i), "float32")) for i in params]
param_input_tuple = tuple_object(params_list)

x_input, y_input = next(iter(train_loader))
x_input = tvm.nd.array(x_input)
y_input = tvm.nd.array(y_input)


epochs = 5
log_interval = 200

# forward and find the gradient
loss, param_grad_tuple = vm["loss_adjoint"](*param_input_tuple, x_input, y_input)
# update parameters
param_input_tuple, opt.state = vm["SGD"](param_input_tuple, param_grad_tuple, opt.state)

print(loss.numpy)
print(len(param_input_tuple), len(param_grad_tuple))
print(param_input_tuple[0])
print(param_grad_tuple[0])

# setup_trainer = SetupTrainer(
#     CrossEntropyLoss(reduction="sum"),
#     SGD(0.01, weight_decay=0.01),
#     # Adam(0.001),
#     [out_sinfo, label_sinfo],
# )

# trainer = Trainer(Backbone, 6, setup_trainer)
# trainer.build(target="llvm", device=tvm.cpu(0))
# trainer.xaiver_uniform_init_params()


# epochs = 5
# log_interval = 200


# for epoch in range(epochs):
#     for batch_idx, (data, target) in enumerate(train_loader):
#         loss = trainer.update_params(data.numpy(), target.numpy())

#         if batch_idx % log_interval == 0 or batch_idx == len(train_loader):
#             print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
#                 f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.numpy():.2f}")

#     total, correct = 0, 0
#     for data, target in test_loader:
#         predict = trainer.predict(data.numpy()) # batch_size * 10
#         total += len(data)
#         correct += np.sum(predict.numpy().argmax(1) == target.numpy().argmax(1))

#     print(f"Train Epoch: {epoch} Accuracy on test dataset: {100.0 * correct / total:.2f}%")



# # # AD process, differentiate "main" and generate a new function "main_adjoint"
# # mod = relax.transform.Gradient(mod.get_global_var("main"), x)(mod)

# # # Show the complete IRModule
# # mod.show()

# # # Optimizer function generation
# # # Note that `opt.state` would be used later to get the state of the optimizer

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
