import tvm
from tvm import relax
from tvm._ffi.registry import get_global_func
from tvm.meta_schedule.relax_integration import tune_relax
from tvm.meta_schedule.runner.runner import Runner
from tvm.meta_schedule.runner.utils import alloc_argument_common
from tvm.runtime.ndarray import NDArray
from tvm.script.parser import relax as R
from tvm.relax.transform.tuning_api import Trace
import numpy as np
@tvm.script.ir_module
class InputModule:
    @R.function
    def main(
        x: R.Tensor((32, 10), "float32"), y: R.Tensor((32,), "int64")
    ):
        with R.dataflow():
            z1 = R.nn.nll_loss(x, y)
            R.output(z1)
        return z1

target, dev = tvm.target.Target("nvidia/geforce-rtx-3080"), tvm.cuda()

mod = relax.transform.LegalizeOps()(InputModule)
mod.show()

work_dir = "/home/yxdong/relax-mlcai/other-repos/AD-Example/tmp/tune1"
def random_fill(data: NDArray):
    random_fill_for_measure = get_global_func("tvm.contrib.random.random_fill_for_measure")
    if 'int' in data.dtype:
        new_data = np.zeros(data.shape, dtype=data.dtype)
        data.copyfrom(new_data)
    else:
        random_fill_for_measure(data)

def alloc_argument(device, args_info, alloc_repeat):
    return alloc_argument_common(random_fill, device, args_info, alloc_repeat)

runner = Runner.create("local", f_alloc_argument=alloc_argument)

tune_relax(mod, {}, target, work_dir, 80, runner=runner)
# tune_relax(mod, {}, target, work_dir, 80)
# with tempfile.TemporaryDirectory() as work_dir:
with target, tvm.transform.PassContext(trace=Trace(mod)):
    mod_new = tvm.transform.Sequential(
        [
            # relax.transform.MetaScheduleTuneIRMod(
            #     params={}, work_dir=work_dir, max_trials_global=80
            # ),
            relax.transform.MetaScheduleApplyDatabase(work_dir),
        ]
    )(mod)
assert relax.analysis.well_formed(mod_new)
mod_new.without_attr("external_mods").show()
