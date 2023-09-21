import tvm
from tvm import relax, tir, te, topi
from tvm.ir.module import IRModule
from tvm.relax.analysis import estimate_memory_usage
from tvm.relax.block_builder import BlockBuilder
from tvm.relay import GlobalVar
from tvm.target.target import Target
import tvm.testing
from tvm.script.parser import relax as R, tir as T, ir as I
import pytest
import numpy as np

from tvm.relax.transform.tuning_api import Trace
from tvm.tir.schedule.schedule import Schedule

from tvm.relax.dpl.pattern import is_op, wildcard


bb = relax.BlockBuilder()
dtype = "float32"
x = relax.Var("x", R.Tensor((1, 2, 5, 5), dtype))
w = relax.Var("w", R.Tensor((2, 2, 1, 1), dtype))


@I.ir_module
class mod:
    @R.function
    def func(x: R.Tensor((3, 3), "float32")):
        x = x + R.const(1, "float32")
        x = x + R.const(1, "float32")
        return x

# mod = bb.get()
mod.show(None, False)

# mod = relax.transform.Gradient("main")(mod)
# mod.show(None, False)
# mod = relax.transform.LegalizeOps()(mod)
# mod = relax.get_pipeline()(mod)
# mod.show(None, False)
# # assert relax.analysis.well_formed(mod)

# target, dev = tvm.target.Target("nvidia/geforce-rtx-3080"), tvm.cuda()
# work_dir = "/home/yxdong/relax-mlcai/other-repos/AD-Example/tmp/tune"
# # with tempfile.TemporaryDirectory() as work_dir:
# with target, tvm.transform.PassContext(trace=Trace(mod)):
#     mod = tvm.transform.Sequential(
#         [
#             relax.transform.MetaScheduleTuneIRMod(
#                 params={}, work_dir=work_dir, max_trials_global=8
#             ),
#             relax.transform.MetaScheduleApplyDatabase(work_dir),
#         ]
#     )(mod)
# assert relax.analysis.well_formed(mod)
# mod.show(None, False)

# with target, tvm.transform.PassContext(trace=Trace(mod)):
#     mod = tvm.tir.transform.DefaultGPUSchedule()(mod)
# mod.show()
# target, dev = "llvm", tvm.cpu()
# ex = relax.build(mod, target)
# vm = relax.VirtualMachine(ex, dev)
# input = tvm.nd.array(np.zeros((2, 2)).astype(np.float64), dev)
# # input1 = tvm.nd.array(np.zeros(()).astype(np.float64), dev)

# res = vm["backbone"](input)
# print(res.numpy().__repr__(), type(res))
