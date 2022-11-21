from __future__ import annotations

from tvm.script._parser import ir as I, relax as R, tir as T
from tvm.ir.base import assert_structural_equal

@I.ir_module
class A:
    @R.function
    def main(x: R.Tensor((1, 5), "float32")):
        with R.dataflow():
            lv1 = x
            R.output(lv1)
        return lv1

@I.ir_module
class B:
    @R.function
    def main1(x: R.Tensor((1, 5), "float32")):
        with R.dataflow():
            lv1 = x
            R.output(lv1)
        return lv1

assert_structural_equal(A["main"], B["main1"])
