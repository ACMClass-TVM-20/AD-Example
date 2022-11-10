@tvm.script.ir_module
class Module:
    @R.function
    def main(x1: Tensor((1, 10), "float32"), y1: Tensor((1, 10), "float32"), x2: Tensor((1, 10), "float32"), y2: Tensor((1, 10), "float32"), z: Tensor((1, 10), "float32")) -> Tuple(Tensor(None, "float32", ndim
 = 0), Tuple(Tensor(None, "float32", ndim = 2), Tensor(None, "float32", ndim = 2), Tensor(None, "float32", ndim = 2), Tensor(None, "float32", ndim = 2), Tensor(None, "float32", ndim = 2))):
        # block 0
        with R.dataflow():
            t: Tuple() = ((x1, y1), (x2, y2))
            lv: Tensor((1, 10), "float32") = t[0][0]
            lv1: Tensor((1, 10), "float32") = t[0][1]
            lv11: Tensor((1, 10), "float32") = relax.add(lv, lv1)
            lv2: Tensor((1, 10), "float32") = t[1][1]
            lv21: Tensor((1, 10), "float32") = relax.sub(lv2, lv11)
            lv3: Tensor((1, 10), "float32") = t[1][0]
            lv31: Tensor((1, 10), "float32") = relax.multiply(lv21, lv3)
            loss: Tensor((), "float32") = relax.nn.softmax_cross_entropy(lv31, z)
            loss_adjoint: Tensor((), "float32") = relax.ones_like(loss)
            lv4: Tensor((1, 10), "float32") = relax.nn.softmax(lv31)
            lv12: Tensor((1, 10), "float32") = relax.sub(lv4, z)
            lv3_adjoint: Tensor((1, 10), "float32") = relax.multiply(loss_adjoint, lv12)
            lv22: Tensor((1, 10), "float32") = relax.multiply(lv3_adjoint, lv21)
            lv3_adjoint1: Tensor((1, 10), "float32") = relax.collapse_sum_like(lv22, lv3)
            lv32: Tensor((1, 10), "float32") = relax.multiply(lv3_adjoint, lv3)
            lv2_adjoint: Tensor((1, 10), "float32") = relax.collapse_sum_like(lv32, lv21)
            lv2_adjoint1: Tensor((1, 10), "float32") = relax.collapse_sum_like(lv2_adjoint, lv2)
            lv41: Tensor((1, 10), "float32") = relax.negative(lv2_adjoint)
            lv1_adjoint: Tensor((1, 10), "float32") = relax.collapse_sum_like(lv41, lv11)
            lv1_adjoint1: Tensor((1, 10), "float32") = relax.collapse_sum_like(lv1_adjoint, lv1)
            lv_adjoint: Tensor((1, 10), "float32") = relax.collapse_sum_like(lv1_adjoint, lv)
            t_adjoint: Tuple() = ((lv_adjoint, lv3_adjoint1), (lv1_adjoint1, lv2_adjoint1))
            x1_adjoint: Tensor((1, 10), "float32") = t_adjoint[0][0]
            y1_adjoint: Tensor((1, 10), "float32") = t_adjoint[0][1]
            x2_adjoint: Tensor((1, 10), "float32") = t_adjoint[1][0]
            y2_adjoint: Tensor((1, 10), "float32") = t_adjoint[1][1]
            lv5: Tensor((1, 10), "float32") = relax.log(lv4)
            lv6: Tensor((1, 10), "float32") = relax.negative(lv5)
            z_adjoint: Tensor((1, 10), "float32") = relax.multiply(loss_adjoint, lv6)
            R.output(loss, x1_adjoint, y1_adjoint, x2_adjoint, y2_adjoint, z_adjoint)
        return (loss, (x1_adjoint, y1_adjoint, x2_adjoint, y2_adjoint, z_adjoint))

