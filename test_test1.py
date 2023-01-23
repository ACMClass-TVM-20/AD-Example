from tvm.script.parser import ir as I, relax as R, tir as T
from tvm.ir.base import assert_structural_equal

@I.ir_module
class B1:
    @R.function
    def main(x: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32"))):
        with R.dataflow():
            lv1 = x[0]
            R.output(lv1)
        return lv1

@I.ir_module
class B2:
    @R.function
    def main(x: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32"))):
        with R.dataflow():
            lv1 = x[0]
            R.output(lv1)
        return lv1

x1 = B1["main"].params[0]
x2 = B2["main"].params[0]

assert_structural_equal([x1.shape, x1.shape], [x1.shape, x2.shape], True)
