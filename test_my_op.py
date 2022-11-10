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
from tvm import relax as rx
from tvm.ir.base import assert_structural_equal
from tvm.relay.testing import rand
# from tvm.script import relax as R
from tvm.testing import assert_allclose
from tvm.testing.utils import check_numerical_grads
from tvm.script._parser import ir as I, relax as R, tir as T

import _gradient
from utils import LowerToTensorIRPass

@I.ir_module
class Before:
    @R.function
    def main(x: R.Tensor((5, 5), "float32"),
                y: R.Tensor((5, 5), "float32"),
                z: R.Tensor((5, 5), "float32"),
                u: R.Tensor((5, 5), "float32")):
        with R.dataflow():
            lv1 = R.add(x, y)
            lv2 = R.subtract(z, u)
            lv3 = R.add(y, z)
            lv4 = R.add(lv1, lv2)
            lv5 = R.add(lv4, lv3)
            lv6 = R.sum(lv5)
            R.output(lv6)
        return lv6

@I.ir_module
class Expected1:
    @R.function
    def main(x: R.Tensor((5, 5), "float32"),
                y: R.Tensor((5, 5), "float32"),
                z: R.Tensor((5, 5), "float32"),
                u: R.Tensor((5, 5), "float32")):
        with R.dataflow():
            lv1 = R.add(x, y)
            lv2 = R.subtract(z, u)
            lv3 = R.add(y, z)
            lv4 = R.add(lv1, lv2)
            lv5 = R.add(lv4, lv3)
            lv6 = R.sum(lv5)
            R.output(lv6)
        return lv6
    
    @R.function
    def main_adjoint(x: R.Tensor((5, 5), "float32"),
                y: R.Tensor((5, 5), "float32"),
                z: R.Tensor((5, 5), "float32"),
                u: R.Tensor((5, 5), "float32")):
        with R.dataflow():
            lv1 = R.add(x, y)
            lv2 = R.subtract(z, u)
            lv3 = R.add(y, z)
            lv4 = R.add(lv1, lv2)
            lv5 = R.add(lv4, lv3)
            lv6 = R.sum(lv5)
            lv6_adjoint = R.ones_like(lv6)
            lv = R.ones_like(lv5)
            lv5_adjoint = R.multiply(lv6_adjoint, lv)
            lv4_adjoint = R.collapse_sum_like(lv5_adjoint, lv4)
            lv3_adjoint = R.collapse_sum_like(lv5_adjoint, lv3)
            lv2_adjoint = R.collapse_sum_like(lv4_adjoint, lv2)
            lv1_adjoint = R.collapse_sum_like(lv4_adjoint, lv1)
            x_adjoint = R.collapse_sum_like(lv1_adjoint, x)
            lv11 = R.collapse_sum_like(lv3_adjoint, y)
            lv21 = R.collapse_sum_like(lv1_adjoint, y)
            y_adjoint = R.add(lv11, lv21)
            lv31 = R.collapse_sum_like(lv3_adjoint, z)
            lv41 = R.collapse_sum_like(lv2_adjoint, z)
            z_adjoint = R.add(lv31, lv41)
            lv51 = R.negative(lv2_adjoint)
            u_adjoint = R.collapse_sum_like(lv51, u)
            R.output(lv6, x_adjoint, y_adjoint, z_adjoint, u_adjoint)
        return relax.Tuple( (lv6, relax.Tuple( (x_adjoint, y_adjoint, z_adjoint, u_adjoint) )) )

After1 = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)

assert_structural_equal(After1["main_adjoint"], Expected1["main_adjoint"])

# print(After1["main_adjoint"].body.shape.fields)
# print(Expected1["main_adjoint"].body.shape.fields)
# print(After1["main_adjoint"].body.shape)
# print(Expected1["main"].body.shape)
# print(Expected1["main_adjoint"].body.shape)