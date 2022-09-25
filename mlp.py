from __future__ import annotations
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm import topi, relax, te
from tvm.script import tir as T
from tvm.script import relax as R

import torch
import torchvision
import matplotlib.pyplot as plt
import pickle as pkl

test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=torchvision.transforms.ToTensor()
)

loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Dataset loaded finish.")

with open("fasionmnist_mlp_params.pkl", "rb") as fp:
    mlp_params = pkl.load(fp)

mlp_params["w0"] = mlp_params["w0"].T
mlp_params["w1"] = mlp_params["w1"].T

print("Build mlp")

from utils import LowerToTensorIRPass

print("TVM version: ", tvm.__version__)

"""
    model
"""

@tvm.script.ir_module
class MultiLayerPerceptron:
    @R.function
    def main(x: Tensor((1, 784), "float32"),
             w0: Tensor((784, 128), "float32"),
             b0: Tensor((128,), "float32"),
             w1: Tensor((128, 10), "float32"),
             b1: Tensor((10,), "float32")) -> Tensor(None, "float32", ndim=2):
        
        # block 0
        with R.dataflow():
            # linear0
            lv0 = relax.matmul(x, w0)
            lv1 = relax.add(lv0, b0)
            # relu0
            lv2 = relax.nn.relu(lv1)
            # linear1
            lv3 = relax.matmul(lv2, w1)
            out = relax.add(lv3, b1)
            R.output(out)
        return out



@tvm.script.ir_module
class AutoDiffMLP:
    @R.function
    def main(x: Tensor((1, 784), "float32"),
             w0: Tensor((784, 128), "float32"),
             b0: Tensor((128,), "float32"),
             w1: Tensor((128, 10), "float32"),
             b1: Tensor((10,), "float32"),
             label: Tensor((1,10), "float32")):
        
        # block 0
        with R.dataflow():
            # linear0
            lv0 = relax.matmul(x, w0)
            lv1 = relax.add(lv0, b0)
            # relu0
            lv2 = relax.nn.relu(lv1)
            # linear1
            lv3 = relax.matmul(lv2, w1)
            out = relax.add(lv3, b1)
            lv4 = relax.nn.softmax(out)
            loss = relax.nn.crossent(lv4, label)
            R.output(out, loss)
        return out, loss


TIRModule = LowerToTensorIRPass()(AutoDiffMLP)
TIRModule.show()

# build and run
ex = relax.vm.build(TIRModule, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

"""
    train
"""

"""
    test
nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}
img, label = next(iter(loader))
data_nd = tvm.nd.array(img.reshape(1, 784))
output = vm["main"](data_nd, nd_params["w0"], nd_params["b0"], nd_params["w1"], nd_params["b1"])
pred_kind = np.argmax(output.numpy(), axis=1)
print("Test Predict: ", class_names[pred_kind[0]])
print("True: ", class_names[label[0]])
"""

nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}
success, total = 0, 0

for img, label in loader:
    data_nd = tvm.nd.array(img.reshape(1, 784))
    label_nd = tvm.nd.array(np.array([[1 if i == label[0] else 0 for i in range(10)]]))
    print(label_nd.shape)
    output, loss = vm["main"](data_nd, nd_params["w0"], nd_params["b0"], nd_params["w1"], nd_params["b1"], label_nd)
    pred_kind = np.argmax(output.numpy(), axis=1)
    total += 1
    if pred_kind[0] == label[0]:
        success += 1
    print("output:", output)
    print("loss:", loss)
    break

print("Prediction Rate: ", float(success)/float(total))