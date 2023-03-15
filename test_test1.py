import numpy as np
import tvm
import tvm.script
import tvm.testing
from tvm import relax
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T
from tvm.relax import training


@I.ir_module
class Backbone:
    I.module_attrs({"params_num": 1, "states_num": 1})
    @R.function
    def predict(x: R.Tensor((2, 2), "float64"), y: R.Tensor((2, 2), "float64"), z: R.Tensor((2, 2), "float64")):
        with R.dataflow():
            x1 = x + y
            z1 = z + R.const(1, "float64")
            R.output(x1, z1)
        return x1, z1

sinfo = relax.TensorStructInfo((2, 2), "float64")

setup_trainer = training.SetupTrainer(
    training.loss.MSELoss(reduction="sum"),
    training.optimizer.SGD(0.001),
    [sinfo, sinfo],
)

train_mod = setup_trainer(Backbone)

trainer = training.Trainer(train_mod)
trainer.build(target="llvm")
# trainer.xaiver_uniform_init_params()
input = np.ones((2, 2))
label = input * 2
res1 = trainer.predict(input)
# print(res1)
res2 = trainer.update_params([input], [label])

print(res2)
