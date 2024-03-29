from __future__ import annotations
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm import topi, relax, te
from tvm.runtime.container import tuple_object
from tvm.script.parser import ir as I, relax as R, tir as T
from tvm.relax.testing import dump_ast
import tvm.script
import _gradient
from tvm.relax.training import Optimizer, SGD, MomentumSGD

from tvm.ir.base import assert_structural_equal
import torch
import torchvision
# import matplotlib.pyplot as plt
import pickle as pkl
np.random.seed(1)
test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Dataset loaded finish.")

"""
with open("fasionmnist_mlp_params.pkl", "rb") as fp:
    mlp_params = pkl.load(fp)
mlp_params["w0"] = mlp_params["w0"].T
mlp_params["w1"] = mlp_params["w1"].T
"""

import math
a0 = math.sqrt(6.0 / (784 + 128))
a1 = math.sqrt(6.0 / (128+10))
mlp_params = [
(a0 * np.random.uniform(-1.0, 1.0, (784, 128))).astype(np.float32),
np.random.uniform(-1.0, 1.0, (128,)).astype(np.float32),
(a1 * np.random.uniform(-1.0, 1.0, (128, 10))).astype(np.float32),
np.random.uniform(-1.0, 1.0, (10,)).astype(np.float32)
]


# print("Build mlp")

from utils import LowerToTensorIRPass

# print("TVM version: ", tvm.__version__)

"""
    model
"""

@I.ir_module
class MultiLayerPerceptron:
    @R.function
    def main(w0: R.Tensor((784, 128), "float32"),
             b0: R.Tensor((128,), "float32"),
             w1: R.Tensor((128, 10), "float32"),
             b1: R.Tensor((10,), "float32"),
             x: R.Tensor((1, 784), "float32"),
             label: R.Tensor((1,10), "float32")):
        # block 0
        with R.dataflow():
            # linear0
            lv0 = R.matmul(x, w0)
            lv1 = R.add(lv0, b0)
            # relu0
            lv2 = R.relu(lv1)
            # linear1
            lv3 = R.matmul(lv2, w1)
            out = R.add(lv3, b1)
            loss = R.softmax_cross_entropy(out, label)
            R.output(loss)
        return loss

# MultiLayerPerceptron.show()

# print(dump_ast(MultiLayerPerceptron["main"]))

ad_var = MultiLayerPerceptron["main"].params[:-2]
AutoDiffMLP = relax.transform.Gradient(MultiLayerPerceptron.get_global_var("main"), require_grads=ad_var)(MultiLayerPerceptron)
AutoDiffMLP.show()

param_list = AutoDiffMLP["main"].params[:-2]
lr = 0.001
opt = SGD(param_list, lr)
# opt = MomentumSGD(param_list, lr, 0.9, 0.1, 0.001, True)
AutoDiffMLP["SGD"] = opt.get_function()
AutoDiffMLP.show()

# # assert_structural_equal(AutoDiffMLP["main_adjoint"], Expected["main_adjoint"])
# TIRModule = LowerToTensorIRPass()(AutoDiffMLP)
# TIRModule.show()

# # # build and run
# ex = relax.vm.build(TIRModule, target="llvm")
# vm = relax.VirtualMachine(ex, tvm.cpu())

# """
#     train
# """

# success, total = 0, 0
batch_size = 64
total_loss = 0
epoch = 0
tvm_params = tuple_object([tvm.nd.array(v) for v in mlp_params])
for img, label in loader:
    data_nd = tvm.nd.array(img.reshape(1, 784))
    label_nd = tvm.nd.array(np.array([[1 if i == label[0] else 0 for i in range(10)]]).astype(np.float32))
    loss, grads = vm["main_adjoint"](*tvm_params, data_nd, label_nd)
    tvm_params, opt.state = vm["MomentumSGD"](tvm_params, grads, opt.state)

    epoch += 1
    total_loss += loss.numpy()

    if epoch % batch_size == 0:
        print("epoch={}, loss={}".format(epoch, total_loss))
        total_loss = 0
