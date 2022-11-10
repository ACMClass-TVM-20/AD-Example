@tvm.script.ir_module
class Module:
    @R.function
    def main(x1: Tensor((1, 10), "float32"), y1: Tensor((1, 10), "float32"), x2: Tensor((1, 10), "float32"), y2: Tensor((1, 10), "float32"), z: Tensor((1, 10), "float32")) -> Tuple(Tensor(None, "float32", ndim
 = 0), Tuple(Tensor(None, "float32", ndim = 2), Tensor(None, "float32", ndim = 2), Tensor(None, "float32", ndim = 2), Tensor(None, "float32", ndim = 2), Tensor(None, "float32", ndim = 2))):
        # block 0
        with R.dataflow():
            lv1: Tensor((1, 10), "float32") = relax.add(x1, y1)
            lv2: Tensor((1, 10), "float32") = relax.sub(y2, lv1)
            lv3: Tensor((1, 10), "float32") = relax.multiply(lv2, x2)
            loss: Tensor((), "float32") = relax.nn.softmax_cross_entropy(lv3, z)
            loss_adjoint: Tensor((), "float32") = relax.ones_like(loss)
            lv: Tensor((1, 10), "float32") = relax.nn.softmax(lv3)
            lv11: Tensor((1, 10), "float32") = relax.sub(lv, z)
            lv3_adjoint: Tensor((1, 10), "float32") = relax.multiply(loss_adjoint, lv11)
            lv21: Tensor((1, 10), "float32") = relax.multiply(lv3_adjoint, x2)
            lv2_adjoint: Tensor((1, 10), "float32") = relax.collapse_sum_like(lv21, lv2)
            lv31: Tensor((1, 10), "float32") = relax.negative(lv2_adjoint)
            lv1_adjoint: Tensor((1, 10), "float32") = relax.collapse_sum_like(lv31, lv1)
            x1_adjoint: Tensor((1, 10), "float32") = relax.collapse_sum_like(lv1_adjoint, x1)
            y1_adjoint: Tensor((1, 10), "float32") = relax.collapse_sum_like(lv1_adjoint, y1)
            lv4: Tensor((1, 10), "float32") = relax.multiply(lv3_adjoint, lv2)
            x2_adjoint: Tensor((1, 10), "float32") = relax.collapse_sum_like(lv4, x2)
            y2_adjoint: Tensor((1, 10), "float32") = relax.collapse_sum_like(lv2_adjoint, y2)
            lv5: Tensor((1, 10), "float32") = relax.log(lv)
            lv6: Tensor((1, 10), "float32") = relax.negative(lv5)
            z_adjoint: Tensor((1, 10), "float32") = relax.multiply(loss_adjoint, lv6)
            R.output(loss, x1_adjoint, y1_adjoint, x2_adjoint, y2_adjoint, z_adjoint)
        return (loss, (x1_adjoint, y1_adjoint, x2_adjoint, y2_adjoint, z_adjoint))