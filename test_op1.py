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
from __future__ import annotations  # must import to defer parsing of annotations

import numpy as np
import pytest
import tvm
import tvm.script
from tvm import relax
from tvm.ir.base import assert_structural_equal
from tvm.relay.testing import rand
# from tvm.script import relax as R
from tvm.testing import assert_allclose
from tvm.testing.utils import check_numerical_grads
from tvm.script._parser import ir as I, relax as R, tir as T

import _gradient
from utils import LowerToTensorIRPass

@I.ir_module
class Test:
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

label = relax.Var("label", [1, 10], relax.DynTensorType(dtype="float32"))

loss = relax.Var("loss", [], relax.DynTensorType(dtype="float32"))

from tvm.ir.op import Op

Test1 = relax.transform.AppendCall(func=Test.get_global_var("main"), op=Op.get("relax.nn.softmax_cross_entropy"), out=loss, args=[Test["main"].body.body, label])(Test)
