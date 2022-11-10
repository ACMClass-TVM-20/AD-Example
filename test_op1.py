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
class Test:
    @R.function
    def main(x: R.Tensor((5, 5), "float32"),
            y: R.Tensor((5, 5), "float32")):
        with R.dataflow():
            lv1 = x
            lv2 = y
            R.output(lv1, lv2)
        return relax.Tuple((lv1, lv2))

print(Test["main"].ret_shape.fields)