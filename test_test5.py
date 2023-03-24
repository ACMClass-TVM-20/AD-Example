import tvm
from tvm import relax
from tvm.script.parser import relax as R

@tvm.script.ir_module
class InputModule:
    @R.function
    def main(
        x: R.Tensor((16, 16), "float32"), y: R.Tensor((16, 16), "float32")
    ) -> R.Tensor((16, 16), "float32"):
        with R.dataflow():
            z = R.multiply(x, y)
            R.output(z)
        return z

# Define your target hardware and device.
target, dev = tvm.target.Target("nvidia/geforce-rtx-3080"), tvm.cuda()

import tempfile
from tvm.relax.transform.tuning_api import Trace

# Run Codegen pass
# with tempfile.TemporaryDirectory() as work_dir:
work_dir = "/home/yxdong/relax-mlcai/other-repos/AD-Example/tune_tmp"
with target, tvm.transform.PassContext(trace=Trace(InputModule)):
    mod = tvm.transform.Sequential(
        [
            relax.transform.LegalizeOps(),
            relax.transform.MetaScheduleTuneIRMod(
                params={}, work_dir=work_dir, max_trials_global=8000
            ),
            relax.transform.MetaScheduleApplyDatabase(work_dir),
        ]
    )(InputModule)
mod.show()
