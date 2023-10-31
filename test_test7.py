from tvm.script import tir as T, ir as I, relax as R
import tvm

@I.ir_module
class mod:
    @R.function
    def main(input: R.Tensor((3,), "float32"), input1: R.Tensor((2, 3, 4), "float32")):
        with R.dataflow():
            # gv = R.reshape(input, (9,))
            gv = R.matmul(input, input1)
            R.output(gv)
        return gv
mod.show()
