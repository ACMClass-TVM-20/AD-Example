@R.function
def main(
    params: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")),
    gradients: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")),
    optim_states: R.Tuple(
        R.Tensor((), dtype="int64"),
        R.Tensor((), dtype="float32"),
        R.Tensor((), dtype="float32"),
        R.Tensor((3, 3), dtype="float32"),
        R.Tensor((3,), dtype="float32"),
        R.Tensor((3, 3), dtype="float32"),
        R.Tensor((3,), dtype="float32"),
    ),
) -> R.Tuple(
    R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")),
    R.Tuple(
        R.Tensor((), dtype="int64"),
        R.Tensor((), dtype="float32"),
        R.Tensor((), dtype="float32"),
        R.Tensor((3, 3), dtype="float32"),
        R.Tensor((3,), dtype="float32"),
        R.Tensor((3, 3), dtype="float32"),
        R.Tensor((3,), dtype="float32"),
    ),
):
    # block 0
    with R.dataflow():
        lv: R.Tensor((3, 3), dtype="float32") = params[0]
        lv1: R.Tensor((3,), dtype="float32") = params[1]
        lv2: R.Tensor((3, 3), dtype="float32") = gradients[0]
        lv3: R.Tensor((3,), dtype="float32") = gradients[1]
        lv4: R.Tensor((), dtype="int64") = optim_states[0]
        lv5: R.Tensor((), dtype="float32") = optim_states[1]
        lv6: R.Tensor((), dtype="float32") = optim_states[2]
        lv7: R.Tensor((3, 3), dtype="float32") = optim_states[3]
        lv8: R.Tensor((3,), dtype="float32") = optim_states[4]
        lv9: R.Tensor((3, 3), dtype="float32") = optim_states[5]
        lv10: R.Tensor((3,), dtype="float32") = optim_states[6]
        lv11: R.Tensor((), dtype="int64") = R.add(lv4, R.const(1, "int64"))
        lv12: R.Tensor((), dtype="float32") = R.multiply(lv5, R.const(0.9, "float32"))
        lv13: R.Tensor((), dtype="float32") = R.multiply(lv6, R.const(0.999, "float32"))
        lv14: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.9, "float32"), lv7)
        lv15: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.1, "float32"), lv2)
        lv16: R.Tensor((3, 3), dtype="float32") = R.add(lv14, lv15)
        lv17: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.999, "float32"), lv9)
        lv18: R.Tensor((3, 3), dtype="float32") = R.multiply(lv2, lv2)
        lv19: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.001, "float32"), lv18)
        lv20: R.Tensor((3, 3), dtype="float32") = R.add(lv17, lv19)
        lv21: R.Tensor((), dtype="float32") = R.subtract(R.const(1, "float32"), lv5)
        lv22: R.Tensor((3, 3), dtype="float32") = R.divide(lv16, lv21)
        lv23: R.Tensor((), dtype="float32") = R.subtract(R.const(1, "float32"), lv6)
        lv24: R.Tensor((3, 3), dtype="float32") = R.divide(lv20, lv23)
        lv25: R.Tensor((3, 3), dtype="float32") = R.sqrt(lv24)
        lv26: R.Tensor((3, 3), dtype="float32") = R.add(lv25, R.const(1e-08, "float32"))
        lv27: R.Tensor((3, 3), dtype="float32") = R.divide(lv22, lv26)
        lv28: R.Tensor((3, 3), dtype="float32") = R.multiply(
            R.const(0.01, "float32"), lv27
        )
        lv29: R.Tensor((3, 3), dtype="float32") = R.subtract(lv, lv28)
        lv30: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.9, "float32"), lv8)
        lv31: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.1, "float32"), lv3)
        lv32: R.Tensor((3,), dtype="float32") = R.add(lv30, lv31)
        lv33: R.Tensor((3,), dtype="float32") = R.multiply(
            R.const(0.999, "float32"), lv10
        )
        lv34: R.Tensor((3,), dtype="float32") = R.multiply(lv3, lv3)
        lv35: R.Tensor((3,), dtype="float32") = R.multiply(
            R.const(0.001, "float32"), lv34
        )
        lv36: R.Tensor((3,), dtype="float32") = R.add(lv33, lv35)
        lv37: R.Tensor((), dtype="float32") = R.subtract(R.const(1, "float32"), lv5)
        lv38: R.Tensor((3,), dtype="float32") = R.divide(lv32, lv37)
        lv39: R.Tensor((), dtype="float32") = R.subtract(R.const(1, "float32"), lv6)
        lv40: R.Tensor((3,), dtype="float32") = R.divide(lv36, lv39)
        lv41: R.Tensor((3,), dtype="float32") = R.sqrt(lv40)
        lv42: R.Tensor((3,), dtype="float32") = R.add(lv41, R.const(1e-08, "float32"))
        lv43: R.Tensor((3,), dtype="float32") = R.divide(lv38, lv42)
        lv44: R.Tensor((3,), dtype="float32") = R.multiply(
            R.const(0.01, "float32"), lv43
        )
        lv45: R.Tensor((3,), dtype="float32") = R.subtract(lv1, lv44)
        gv: R.Tuple(
            R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")
        ) = (lv29, lv45)
        gv1: R.Tuple(
            R.Tensor((), dtype="int64"),
            R.Tensor((), dtype="float32"),
            R.Tensor((), dtype="float32"),
            R.Tensor((3, 3), dtype="float32"),
            R.Tensor((3,), dtype="float32"),
            R.Tensor((3, 3), dtype="float32"),
            R.Tensor((3,), dtype="float32"),
        ) = (lv11, lv12, lv13, lv16, lv32, lv20, lv36)
        R.output(gv, gv1)
    return (gv, gv1)
