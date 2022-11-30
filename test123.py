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
from tvm.testing import assert_allclose
from tvm.testing.utils import check_numerical_grads
from tvm.script._parser import ir as I, relax as R, tir as T
from tvm._ffi.base import TVMError
from tvm.relax.transform import OperatorLegalizer

import tvm.relax.training.legalizer_update


def _execute_mod(mod, func_name, *args):
    # lowered_mod = LowerToTensorIRPass()(mod)
    lowered_mod = OperatorLegalizer(mod).transform()
    ex = relax.vm.build(lowered_mod, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm[func_name](*args)


def test_binding_uses():
    # This case tests:
    # - Different type of bindings: assign binding & call binding;
    # - One def and multiple uses.
    # - Unused variable in module
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((5, 5), "float32"),
                 y: R.Tensor((5,), "float32"),
                 z: R.Tensor((5,), "float32"),
                 u: R.Tensor((5,), "float32")):
            with R.dataflow():
                lv1 = x
                lv2 = R.add(lv1, y)
                lv3 = R.add(lv2, y)
                lv4 = R.add(x, lv3)
                lv5 = lv3
                lv6 = R.add(x, lv5)
                lv7 = R.sum(lv4)
                lv8 = R.add(lv6, z) # unused
                R.output(lv7)
            return lv7
    After = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)

    args = [rand("float32", 5, 5), rand("float32", 5), rand("float32", 5), rand("float32", 5)]
    output, grads = _execute_mod(After, "main_adjoint", *args)
    assert_allclose(output.numpy(), np.sum(2 * args[0].numpy() + 2 * args[1].numpy()), atol=1e-4)
    expected_grads_nd = [2 * np.ones_like(args[0].numpy()),
                         10 * np.ones_like(args[1].numpy()),
                         np.zeros_like(args[2].numpy()),
                         np.zeros_like(args[3].numpy())]

    for i, j in zip(grads, expected_grads_nd):
        assert_allclose(i.numpy(), j)

test_binding_uses()
