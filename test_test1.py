import numpy as np
import numpy.random
import pytest
import tvm
import tvm.script
import tvm.testing
from tvm import relax
from tvm import relax as rx
from tvm import te, tir
from tvm.ir.base import assert_structural_equal
from tvm.relax import Function, Var
from tvm.relax.block_builder import BlockBuilder
from tvm.relax.expr import Expr
from tvm.relax.op import add, divide, multiply, sqrt, subtract
from tvm.relax.struct_info import TensorStructInfo
from tvm.relax.transform import LegalizeOps
from tvm.runtime.container import tuple_object
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T
from tvm.tir.expr import IntImm

@tvm.script.ir_module
class Module:
    @R.function
    def main(
        grad: R.Tensor((3, 2, 8, 8), dtype="float32"),
        x: R.Tensor((3, 2, 10, 10), dtype="float32"),
    ) -> R.Tuple(
        R.Tensor((3, 2, 8, 8), dtype="float32"),
        R.Tensor((3, 2, 10, 10), dtype="float32"),
    ):
        # block 0
        gv = R.call_tir(
            max_pool2d, (x,), out_sinfo=R.Tensor((3, 2, 8, 8), dtype="float32")
        )
        gv1 = R.call_tir(
            max_pool2d_backward,
            (grad, x),
            out_sinfo=R.Tensor((3, 2, 10, 10), dtype="float32"),
        )
        return (gv, gv1)

    @T.prim_func
    def max_pool2d(
        rxplaceholder: T.Buffer[
            (T.int64(3), T.int64(2), T.int64(10), T.int64(10)), "float32"
        ],
        pool_max: T.Buffer[(T.int64(3), T.int64(2), T.int64(8), T.int64(8)), "float32"],
    ):
        # function attr dict
        T.func_attr({"tir.noalias": True})
        # body
        # with T.block("root")
        for ax0, ax1, ax2, ax3, rv0, rv1 in T.grid(
            T.int64(3), T.int64(2), T.int64(8), T.int64(8), T.int64(3), T.int64(3)
        ):
            with T.block("pool_max"):
                v_ax0, v_ax1, v_ax2, v_ax3, v_rv0, v_rv1 = T.axis.remap(
                    "SSSSRR", [ax0, ax1, ax2, ax3, rv0, rv1]
                )
                T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2 + v_rv0, v_ax3 + v_rv1])
                T.writes(pool_max[v_ax0, v_ax1, v_ax2, v_ax3])
                T.block_attr({"schedule_rule": "meta_schedule.pool_max"})
                with T.init():
                    pool_max[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(
                        -3.4028234663852886e38
                    )
                pool_max[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(
                    pool_max[v_ax0, v_ax1, v_ax2, v_ax3],
                    rxplaceholder[v_ax0, v_ax1, v_ax2 + v_rv0, v_ax3 + v_rv1],
                )

    @T.prim_func
    def max_pool2d_backward(
        output_grad: T.Buffer[
            (T.int64(3), T.int64(2), T.int64(8), T.int64(8)), "float32"
        ],
        input: T.Buffer[
            (T.int64(3), T.int64(2), T.int64(10), T.int64(10)), "float32"
        ],
        input_grad: T.Buffer[
            (T.int64(3), T.int64(2), T.int64(10), T.int64(10)), "float32"
        ],
    ):
        # function attr dict
        T.func_attr({"tir.noalias": True})
        # body
        # with T.block("root")
        extracted_tensor_v0 = T.alloc_buffer(
            [T.int64(8), T.int64(8), T.int64(3), T.int64(2), T.int64(10), T.int64(10)],
            dtype="float32",
        )
        extracted_tensor_v1 = T.alloc_buffer(
            [T.int64(8), T.int64(8), T.int64(3), T.int64(2), T.int64(10), T.int64(10)],
            dtype="float32",
        )
        for (
            index8,
            n1_n1_ax3_shifted_shifted,
            n2_n2_shifted_shifted,
            n3_n3_shifted_shifted,
            n4_n4_jac_i2_shifted_shifted,
            n5_n5_jac_i3_shifted_shifted,
            n0_n0_rv0_shifted_shifted,
            n1_n1_rv1_shifted_shifted,
        ) in T.grid(
            T.int64(8),
            T.int64(8),
            T.int64(3),
            T.int64(2),
            T.int64(10),
            T.int64(10),
            T.int64(3),
            T.int64(3),
        ):
            with T.block("extracted_tensor"):
                (
                    v_n0_n0_ax2_shifted_shifted,
                    v_n1_n1_ax3_shifted_shifted,
                    v_n2_n2_shifted_shifted,
                    v_n3_n3_shifted_shifted,
                    v_n4_n4_jac_i2_shifted_shifted,
                    v_n5_n5_jac_i3_shifted_shifted,
                    v_n0_n0_rv0_shifted_shifted,
                    v_n1_n1_rv1_shifted_shifted,
                ) = T.axis.remap(
                    "SSSSSSRR",
                    [
                        index8,
                        n1_n1_ax3_shifted_shifted,
                        n2_n2_shifted_shifted,
                        n3_n3_shifted_shifted,
                        n4_n4_jac_i2_shifted_shifted,
                        n5_n5_jac_i3_shifted_shifted,
                        n0_n0_rv0_shifted_shifted,
                        n1_n1_rv1_shifted_shifted,
                    ],
                )
                T.reads(
                    input[
                        T.int64(2) - v_n2_n2_shifted_shifted,
                        T.int64(1) - v_n3_n3_shifted_shifted,
                        v_n0_n0_ax2_shifted_shifted + v_n0_n0_rv0_shifted_shifted,
                        v_n1_n1_ax3_shifted_shifted + v_n1_n1_rv1_shifted_shifted,
                    ]
                )
                T.writes(
                    extracted_tensor_v0[
                        v_n0_n0_ax2_shifted_shifted,
                        v_n1_n1_ax3_shifted_shifted,
                        v_n2_n2_shifted_shifted,
                        v_n3_n3_shifted_shifted,
                        v_n4_n4_jac_i2_shifted_shifted,
                        v_n5_n5_jac_i3_shifted_shifted,
                    ],
                    extracted_tensor_v1[
                        v_n0_n0_ax2_shifted_shifted,
                        v_n1_n1_ax3_shifted_shifted,
                        v_n2_n2_shifted_shifted,
                        v_n3_n3_shifted_shifted,
                        v_n4_n4_jac_i2_shifted_shifted,
                        v_n5_n5_jac_i3_shifted_shifted,
                    ],
                )
                with T.init():
                    extracted_tensor_v0[
                        v_n0_n0_ax2_shifted_shifted,
                        v_n1_n1_ax3_shifted_shifted,
                        v_n2_n2_shifted_shifted,
                        v_n3_n3_shifted_shifted,
                        v_n4_n4_jac_i2_shifted_shifted,
                        v_n5_n5_jac_i3_shifted_shifted,
                    ] = T.float32(0)
                    extracted_tensor_v1[
                        v_n0_n0_ax2_shifted_shifted,
                        v_n1_n1_ax3_shifted_shifted,
                        v_n2_n2_shifted_shifted,
                        v_n3_n3_shifted_shifted,
                        v_n4_n4_jac_i2_shifted_shifted,
                        v_n5_n5_jac_i3_shifted_shifted,
                    ] = T.float32(-3.4028234663852886e38)
                v_extracted_tensor_v0: T.float32 = extracted_tensor_v0[
                    v_n0_n0_ax2_shifted_shifted,
                    v_n1_n1_ax3_shifted_shifted,
                    v_n2_n2_shifted_shifted,
                    v_n3_n3_shifted_shifted,
                    v_n4_n4_jac_i2_shifted_shifted,
                    v_n5_n5_jac_i3_shifted_shifted,
                ] * T.Select(
                    input[
                        T.int64(2) - v_n2_n2_shifted_shifted,
                        T.int64(1) - v_n3_n3_shifted_shifted,
                        v_n0_n0_ax2_shifted_shifted + v_n0_n0_rv0_shifted_shifted,
                        v_n1_n1_ax3_shifted_shifted + v_n1_n1_rv1_shifted_shifted,
                    ]
                    <= extracted_tensor_v1[
                        v_n0_n0_ax2_shifted_shifted,
                        v_n1_n1_ax3_shifted_shifted,
                        v_n2_n2_shifted_shifted,
                        v_n3_n3_shifted_shifted,
                        v_n4_n4_jac_i2_shifted_shifted,
                        v_n5_n5_jac_i3_shifted_shifted,
                    ],
                    T.float32(1),
                    T.float32(0),
                ) + T.Select(
                    v_n4_n4_jac_i2_shifted_shifted
                    - v_n0_n0_ax2_shifted_shifted
                    - T.Cast("int64", v_n0_n0_rv0_shifted_shifted)
                    == T.int64(0)
                    and v_n5_n5_jac_i3_shifted_shifted
                    - v_n1_n1_ax3_shifted_shifted
                    - T.Cast("int64", v_n1_n1_rv1_shifted_shifted)
                    == T.int64(0),
                    T.float32(1),
                    T.float32(0),
                ) * T.Select(
                    input[
                        T.int64(2) - v_n2_n2_shifted_shifted,
                        T.int64(1) - v_n3_n3_shifted_shifted,
                        v_n0_n0_ax2_shifted_shifted + v_n0_n0_rv0_shifted_shifted,
                        v_n1_n1_ax3_shifted_shifted + v_n1_n1_rv1_shifted_shifted,
                    ]
                    <= extracted_tensor_v1[
                        v_n0_n0_ax2_shifted_shifted,
                        v_n1_n1_ax3_shifted_shifted,
                        v_n2_n2_shifted_shifted,
                        v_n3_n3_shifted_shifted,
                        v_n4_n4_jac_i2_shifted_shifted,
                        v_n5_n5_jac_i3_shifted_shifted,
                    ],
                    T.float32(0),
                    T.float32(1),
                )
                v_extracted_tensor_v1: T.float32 = T.max(
                    extracted_tensor_v1[
                        v_n0_n0_ax2_shifted_shifted,
                        v_n1_n1_ax3_shifted_shifted,
                        v_n2_n2_shifted_shifted,
                        v_n3_n3_shifted_shifted,
                        v_n4_n4_jac_i2_shifted_shifted,
                        v_n5_n5_jac_i3_shifted_shifted,
                    ],
                    input[
                        T.int64(2) - v_n2_n2_shifted_shifted,
                        T.int64(1) - v_n3_n3_shifted_shifted,
                        v_n0_n0_ax2_shifted_shifted + v_n0_n0_rv0_shifted_shifted,
                        v_n1_n1_ax3_shifted_shifted + v_n1_n1_rv1_shifted_shifted,
                    ],
                )
                extracted_tensor_v0[
                    v_n0_n0_ax2_shifted_shifted,
                    v_n1_n1_ax3_shifted_shifted,
                    v_n2_n2_shifted_shifted,
                    v_n3_n3_shifted_shifted,
                    v_n4_n4_jac_i2_shifted_shifted,
                    v_n5_n5_jac_i3_shifted_shifted,
                ] = v_extracted_tensor_v0
                extracted_tensor_v1[
                    v_n0_n0_ax2_shifted_shifted,
                    v_n1_n1_ax3_shifted_shifted,
                    v_n2_n2_shifted_shifted,
                    v_n3_n3_shifted_shifted,
                    v_n4_n4_jac_i2_shifted_shifted,
                    v_n5_n5_jac_i3_shifted_shifted,
                ] = v_extracted_tensor_v1
        for (
            ax0,
            ax1,
            ax2,
            ax3,
            n0_n0_k2_shifted_shifted,
            n1_n1_k3_shifted_shifted,
        ) in T.grid(
            T.int64(3), T.int64(2), T.int64(10), T.int64(10), T.int64(8), T.int64(8)
        ):
            with T.block("pool_max_rxplaceholder_grad"):
                (
                    v_ax0,
                    v_ax1,
                    v_ax2,
                    v_ax3,
                    v_n0_n0_k2_shifted_shifted,
                    v_n1_n1_k3_shifted_shifted,
                ) = T.axis.remap(
                    "SSSSRR",
                    [
                        ax0,
                        ax1,
                        ax2,
                        ax3,
                        n0_n0_k2_shifted_shifted,
                        n1_n1_k3_shifted_shifted,
                    ],
                )
                T.reads(
                    output_grad[
                        v_ax0,
                        v_ax1,
                        v_n0_n0_k2_shifted_shifted,
                        v_n1_n1_k3_shifted_shifted,
                    ],
                    extracted_tensor_v0[
                        v_n0_n0_k2_shifted_shifted,
                        v_n1_n1_k3_shifted_shifted,
                        T.int64(2) - v_ax0,
                        T.int64(1) - v_ax1,
                        v_ax2,
                        v_ax3,
                    ],
                )
                T.writes(input_grad[v_ax0, v_ax1, v_ax2, v_ax3])
                with T.init():
                    input_grad[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(
                        0
                    )
                input_grad[v_ax0, v_ax1, v_ax2, v_ax3] = (
                    input_grad[v_ax0, v_ax1, v_ax2, v_ax3]
                    + output_grad[
                        v_ax0,
                        v_ax1,
                        v_n0_n0_k2_shifted_shifted,
                        v_n1_n1_k3_shifted_shifted,
                    ]
                    * extracted_tensor_v0[
                        v_n0_n0_k2_shifted_shifted,
                        v_n1_n1_k3_shifted_shifted,
                        T.int64(2) - v_ax0,
                        T.int64(1) - v_ax1,
                        v_ax2,
                        v_ax3,
                    ]
                )


ex = relax.vm.build(Module, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())
u_in = tvm.nd.array(numpy.ones((3, 2, 8, 8)).astype(np.float32))
x_in = tvm.nd.array(numpy.random.rand(3, 2, 10, 10).astype(np.float32))

print("grad:\n", u_in.numpy(), "\nx:\n", x_in.numpy(), sep='')

res = vm["main"](u_in, x_in)
# print(res.numpy())
print("res:\n",res[0].numpy())
print("grad res:\n", res[1].numpy())
