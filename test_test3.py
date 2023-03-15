from typing import Callable, List, Tuple
import numpy as np
import numpy.random
import pytest
import tvm
from tvm._ffi.base import TVMError
from tvm.arith.analyzer import Analyzer
import tvm.script
import tvm.testing
from tvm import relax
from tvm import relax as rx
from tvm import te, tir
from tvm.ir.base import assert_structural_equal
from tvm.relax import Function, Var
from tvm.relax.block_builder import BlockBuilder
from tvm.relax.expr import Expr
from tvm import topi, relay
from tvm.relax.op import add, divide, multiply, sqrt, subtract
from tvm.relax.struct_info import TensorStructInfo
from tvm.relax.transform import LegalizeOps
from tvm.runtime.container import tuple_object
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T
from tvm.tir.function import PrimFunc
from tvm.relax import training
from tvm.relax.frontend.torch import from_fx


'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fx
import numpy.random

dtype = "float64"


class TrainerContext():
    bb: BlockBuilder
    param_list: List[Var]
    p_default: List[np.ndarray]
    state_list: List[Var]
    updated_state_list: List[Var]
    s_default: List[np.ndarray]

    def __init__(self, bb):
        self.bb = bb
        self.param_list = []
        self.p_default = []
        self.state_list = []
        self.updated_state_list = []
        self.s_default = []

    def __getattr__(self, name):
        return getattr(self.bb, name)

    def add_param(self, var, default_val):
        self.param_list.append(var)
        self.p_default.append(default_val)

    def add_state(self, var, updated_var, default_val):
        self.state_list.append(var)
        self.updated_state_list.append(updated_var)
        self.s_default.append(default_val)

    def emit_output_list(self, exprs: List[Expr]):
        ret = []
        for e in exprs:
            ret.append(self.bb.emit_output(e))
        return ret

    def func_params(self):
        return self.param_list + self.state_list

    def mod_attrs(self):
        return {"params_num": len(self.param_list), "states_num": len(self.state_list)}


def get_np_shape(expr):
    return [int(i) for i in expr.struct_info.shape]


def Conv2d(ctx: TrainerContext, input, in_channel, out_channel, kernel_size, stride, padding=0):
    weight = relax.Var("conv2d_weight", R.Tensor((out_channel, in_channel, kernel_size, kernel_size), dtype))

    # kaiming init
    bound = 1.0 / np.sqrt(in_channel * kernel_size * kernel_size)
    ctx.add_param(weight, numpy.random.uniform(-bound, bound, size=get_np_shape(weight)).astype(dtype))

    res = ctx.emit(R.nn.conv2d(input, weight, stride, padding))
    return res


def BatchNorm2d(ctx: TrainerContext, input, channel):
    gamma = relax.Var("bn_gamma", R.Tensor((channel,), dtype))
    beta = relax.Var("bn_beta", R.Tensor((channel,), dtype))
    moving_mean = relax.Var("bn_mm", R.Tensor((channel,), dtype))
    moving_var = relax.Var("bn_mv", R.Tensor((channel,), dtype))

    ctx.add_param(gamma, np.ones(get_np_shape(gamma)).astype(dtype))
    ctx.add_param(beta, np.zeros(get_np_shape(beta)).astype(dtype))

    bn = ctx.emit(R.nn.batch_norm(input, gamma, beta, moving_mean, moving_var))
    res, new_moving_mean, new_moving_var = ctx.emit(bn[0]), ctx.emit(bn[1]), ctx.emit(bn[2])

    ctx.add_state(moving_mean, new_moving_mean, np.zeros(get_np_shape(moving_mean)).astype(dtype))
    ctx.add_state(moving_var, new_moving_var, np.ones(get_np_shape(moving_mean)).astype(dtype))

    return res


def Linear(ctx: TrainerContext, input, in_feature, out_feature):
    weight = relax.Var("ln_weight", R.Tensor((in_feature, out_feature), dtype))
    bias = relax.Var("ln_bias", R.Tensor((out_feature,), dtype))

    bound = 1.0 / np.sqrt(in_feature)
    ctx.add_param(weight, numpy.random.uniform(-bound, bound, size=(get_np_shape(weight))))
    ctx.add_param(bias, numpy.random.uniform(-bound, bound, size=(get_np_shape(bias))))

    res = ctx.emit(R.matmul(input, weight) + bias)
    return res


def BasicBlock(ctx: TrainerContext, input, in_planes, planes, stride=1):
    expansion = 1
    conv1 = Conv2d(ctx, input, in_planes, planes, 3, stride, 1)
    bn1 = BatchNorm2d(ctx, conv1, planes)
    relu1 = ctx.emit(R.nn.relu(bn1))
    conv2 = Conv2d(ctx, relu1, planes, planes, 3, 1, 1)
    bn2 = BatchNorm2d(ctx, conv2, planes)
    shortcut = input
    if stride != 1 or in_planes != expansion * planes:
        conv3 = Conv2d(ctx, input, in_planes, expansion * planes, 1, stride)
        shortcut = BatchNorm2d(ctx, conv3, expansion * planes)
    relu2 = ctx.emit(R.nn.relu(bn2 + shortcut))
    return relu2


def get_expansion(block):
    return 1 if block is BasicBlock else 4


def ResNet_layer(ctx: TrainerContext, input, block, in_planes, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    for stride in strides:
        input = block(ctx, input, in_planes, planes, stride)
        in_planes = planes * get_expansion(block)
    return input, in_planes


def ResNet(ctx: TrainerContext, input, block, num_blocks, num_classes=10):
    in_planes = 64
    conv1 = Conv2d(ctx, input, 3, 64, 3, 1, 1)
    bn1 = BatchNorm2d(ctx, conv1, 64)
    relu1 = ctx.emit(R.nn.relu(bn1))
    layer1, in_planes = ResNet_layer(ctx, relu1, block, in_planes, 64, num_blocks[0], 1)
    layer2, in_planes = ResNet_layer(ctx, layer1, block, in_planes, 128, num_blocks[1], 2)
    layer3, in_planes = ResNet_layer(ctx, layer2, block, in_planes, 256, num_blocks[2], 2)
    layer4, in_planes = ResNet_layer(ctx, layer3, block, in_planes, 512, num_blocks[3], 2)
    pool = ctx.emit(R.nn.avg_pool2d(layer4, 4, 1, 0, ceil_mode=False))
    reshape = ctx.emit(R.reshape(pool, (pool.struct_info.shape[0], -1)))
    linear = Linear(ctx, reshape, 512 * get_expansion(block), num_classes)
    return linear


def ResNet18(ctx: TrainerContext, input):
    return ResNet(ctx, input, BasicBlock, [2, 2, 2, 2])


bb = BlockBuilder()
ctx = TrainerContext(bb)
input = relax.Var("input", R.Tensor((1, 3, 32, 32), "float64"))
input_list = [input]

with bb.function("predict"):
    with bb.dataflow():
        result = ResNet18(ctx, input)
        ret = ctx.emit_output_list([result] + ctx.updated_state_list)
    bb.emit_func_output(ret, input_list + ctx.func_params())

Backbone = bb.get()
Backbone = Backbone.with_attrs(ctx.mod_attrs())

Backbone.show()

out_sinfo = relax.TensorStructInfo((1, 10), "float64")
label_sinfo = relax.TensorStructInfo((1,), "int64")


setup_trainer = training.SetupTrainer(
    training.loss.CrossEntropyLoss(),
    training.optimizer.MomentumSGD(0.1, 0.9, weight_decay=5e-4),
    [out_sinfo, label_sinfo],
)

train_mod = setup_trainer(Backbone)

trainer = training.Trainer(train_mod)
trainer.build(target="llvm")
trainer.load_params(ctx.p_default)
trainer.load_states(ctx.s_default)
input = numpy.random.randint(0, 5, (1, 3, 32, 32)).astype(dtype)
# label = numpy.random.rand()
res1 = trainer.predict(input)
print(res1)
# res2 = trainer.update_params([input], [label])

# print(res2)
