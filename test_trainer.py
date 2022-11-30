# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm
import numpy as np
from tvm.script.parser import ir as I, relax as R, tir as T
from tvm.relax.training import Trainer, SGD, MomentumSGD

@I.ir_module
class MLP:
    @R.function
    def main(x: R.Tensor((1, 784), "float32"),
            w0: R.Tensor((784, 128), "float32"),
            b0: R.Tensor((128,), "float32"),
            w1: R.Tensor((128, 10), "float32"),
            b1: R.Tensor((10,), "float32")):

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
            R.output(out)
        return out

import torch
import torchvision

test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=torchvision.transforms.ToTensor()
)

loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)


trainer1 = Trainer(backbone=MLP,
                func_name="main",
                parameters_indices=range(1, 5)
            )

trainer2 = Trainer(backbone=MLP,
                func_name="main",
                parameters_indices=range(1, 5)
            )

trainer1.prepare("relax.nn.softmax_cross_entropy", SGD(None, 0.01))
trainer2.prepare("relax.nn.softmax_cross_entropy", MomentumSGD(None, 0.001, 0.9, 0.1, 0.001, True))

def trainer_setting_pipeline(trainer):
    trainer.set_vm_config(target="llvm", device=tvm.cpu())
    trainer.setup()
    trainer.rand_init_params()
    return trainer_setting_pipeline

trainer_setting_pipeline(trainer1)(trainer2)

def _hook(dataline):
    return np.array(dataline[0].reshape(1, 784)).astype(np.float32), \
        np.array([[1 if i == dataline[1][0] else 0 for i in range(10)]]).astype(np.float32)

# trainer1.train(epoch=10, loader=loader, data_hook=_hook, show_detail=True)
trainer2.train(epoch=10, loader=loader, data_hook=_hook, show_detail=True)
