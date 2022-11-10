from __future__ import annotations
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm import topi, relax, te
from tvm.script import tir as T
from tvm.script import relax as R
from tvm.relax.testing import dump_ast


@tvm.script.ir_module
class TestShape:
    @R.function
    def main(x: Tensor((3, 4), "float32"),
             y: Tensor((4, 3), "float32"),
             z: Tensor((3, 3), "float32")):
        
        # block 0
        with R.dataflow():
            # linear0
            lv0 = relax.matmul(x, y)
            out = relax.add(lv0, z)
            R.output(out)
            
        return out

TestShape.show()

