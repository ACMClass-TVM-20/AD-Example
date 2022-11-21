from __future__ import annotations
from tvm.script._parser import ir as I, relax as R, tir as T
from utils import LowerToTensorIRPass
from tvm import relax

@I.ir_module
class MLP:
    @R.function
    def main(x: R.Tensor((1, 784), "float32"),
             w0: R.Tensor((784, 128), "float32"),
             b0: R.Tensor((128,), "float32"),
             w1: R.Tensor((128, 10), "float32"),
             b1: R.Tensor((10,), "float32"),
             label: R.Tensor((1,10), "float32")):

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
            loss = R.softmax_cross_entropy(out, label)
            R.output(loss)

        return loss

print("Before: ")
MLP.show()

AutoDiffMLP = relax.transform.SimpleAD(MLP.get_global_var("main"), require_grads=MLP["main"].params[1:5])(MLP)
print("After: ")
AutoDiffMLP.show()

lowered_mod = LowerToTensorIRPass()(AutoDiffMLP)
lowered_mod.show()

