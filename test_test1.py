import tempfile
import numpy as np
import tvm
from tvm.meta_schedule.relax_integration import tune_relax
import tvm.script
import tvm.testing
from tvm import relax
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T
from tvm.relax import training

from tvm.relax.transform.tuning_api import Trace

@I.ir_module
class Backbone:
    I.module_attrs({"param_num": 1, "state_num": 1})
    @R.function
    def backbone(x: R.Tensor((2, 2), "float64"), y: R.Tensor((2, 2), "float64"), z: R.Tensor((2, 2), "float64")):
        with R.dataflow():
            x1 = x + y
            z1 = z + R.const(1, "float64")
            R.output(x1, z1)
        return x1, z1

sinfo = relax.TensorStructInfo((2, 2), "float64")

setup_trainer = training.SetupTrainer(
    training.loss.MSELoss(reduction="sum"),
    training.optimizer.SGD(0.1),
    [sinfo, sinfo],
)

train_mod = setup_trainer(Backbone)
train_mod.without_attr("optim_state").show()
target, dev = tvm.target.Target("nvidia/geforce-rtx-3080"), tvm.cuda()

# with tempfile.TemporaryDirectory() as work_dir:

work_dir = "/home/yxdong/relax-mlcai/other-repos/AD-Example/tmp/tune1"
with target, tvm.transform.PassContext(trace=Trace(train_mod)):
    # tune_relax(train_mod, {}, target, work_dir, 80)
    tuned_mod = relax.transform.MetaScheduleApplyDatabase(work_dir)(train_mod)


ex = relax.build(tuned_mod, target)
vm = relax.VirtualMachine(ex, dev, profile=True)

trainer = training.Trainer(train_mod, vm, dev)
# trainer.build(target="llvm")
# trainer.xaiver_uniform_init_params()
input = np.ones((2, 2))
label = input * 2
res1 = trainer.predict(input)
print(res1)
for i in range(100):
    res2 = trainer.update([input], [label])
print(res2)

res3 = trainer.profile_backward([input], [label])
print(res3.table())
