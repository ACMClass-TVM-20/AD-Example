from typing import Callable, List, Tuple

import numpy as np

# import numpy as np
# import numpy.random
# import pytest
import tvm

# from tvm._ffi.base import TVMError
# from tvm.arith.analyzer import Analyzer
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


import tvm
from tvm import relax
from tvm.script.parser import relax as R
import numpy as np

# Define an example IRModule
@tvm.script.ir_module
class InputModule:
    @R.function
    def main(
        x: R.Tensor((16, 16), "float32"), y: R.Tensor((16, 16), "float32")
    ) -> R.Tensor((16, 16), "float32"):
        with R.dataflow():
            z1 = R.multiply(x, y)
            z2 = R.add(z1, x)
            z3 = R.add(z1, z2)
            z4 = R.multiply(z3, z2)
            z5 = R.add(z4, z1)
            R.output(z5)
        return z5


mod = InputModule

from tvm.relax.dpl import is_op, wildcard

# patterns = [
#         ("tensorrt.multiply", is_op("relax.multiply")(wildcard(), wildcard())),
#         ("tensorrt.add", is_op("relax.add")(wildcard(), wildcard())),
# ]

# from tvm import relax
# mod1 = relax.transform.FuseOpsByPattern(patterns)(mod)
# mod1.show()

# mod2 = relax.transform.MergeCompositeFunctions()(mod1)
# mod2.show()

# mod3 = relax.transform.RunCodegen()(mod2)
# mod3.without_attr("external_mods").show()
# # Produced runtime module will be attached in the IRModule attribute.
# print(f"TensorRT runtime module: {mod3.attrs['external_mods']}")

# # Check if output IRModule is well-formed.
# assert relax.analysis.well_formed(mod3)

# # Define your target hardware and device.
# target, dev = tvm.target.Target("cuda"), tvm.cuda()

# Prepare inputs.
# np0 = np.random.rand(16, 16).astype(np.float32)
# np1 = np.random.rand(16, 16).astype(np.float32)
# data0 = tvm.nd.array(np0, dev)
# data1 = tvm.nd.array(np1, dev)
# inputs = [data0, data1]

# # Prepare expected output.
# t1 = np.multiply(np0, np1)
# t2 = np.add(t1, np0)
# t3 = np.add(t1, t2)
# t4 = np.multiply(t3, t2)
# expected = np.add(t4, t1)

# # Build and prepare VM.
# ex = relax.build(mod3, target, params={})
# vm = relax.VirtualMachine(ex, dev)

# # Run VM.
# out = vm["main"](*inputs)

# import tvm.testing
# tvm.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

patterns = [
    ("tensorrt.add", is_op("relax.add")(wildcard(), wildcard())),
]
# Define your target hardware and device.
target, dev = tvm.target.Target("nvidia/geforce-rtx-3080"), tvm.cuda()

import tempfile
from tvm.relax.transform.tuning_api import Trace

# Run Codegen pass

work_dir = "/home/yxdong/relax-mlcai/other-repos/AD-Example/tune_tmp"
# with tempfile.TemporaryDirectory() as work_dir:
with target, tvm.transform.PassContext(trace=Trace(mod)):
    mod4 = tvm.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern(patterns),
            relax.transform.MergeCompositeFunctions(),
            relax.transform.RunCodegen(),
            relax.transform.LegalizeOps(),
            relax.transform.MetaScheduleTuneIRMod(
                params={}, work_dir=work_dir, max_trials_global=8
            ),
            relax.transform.MetaScheduleApplyDatabase(work_dir),
        ]
    )(mod)
assert relax.analysis.well_formed(mod4)
mod4.without_attr("external_mods").show()

# Build and prepare VM.
ex = relax.build(mod4, target, params={})
vm = relax.VirtualMachine(ex, dev)

np0 = np.random.rand(16, 16).astype(np.float32)
np1 = np.random.rand(16, 16).astype(np.float32)
data0 = tvm.nd.array(np0, dev)
data1 = tvm.nd.array(np1, dev)
inputs = [data0, data1]

# Prepare expected output.
t1 = np.multiply(np0, np1)
t2 = np.add(t1, np0)
t3 = np.add(t1, t2)
t4 = np.multiply(t3, t2)
expected = np.add(t4, t1)

# Run VM.
out = vm["main"](*inputs)
tvm.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)
