from dataclasses import dataclass
from typing import List, Literal

import tvm
from tvm import relax, te, tir, topi
from tvm import tir

from tvm.script import tir as T

"""TIR computation utilities for quantization."""


# fmt: off
def _tir_f32x2_to_bf16x2_to_u32(v0: tir.PrimExpr, v1: tir.PrimExpr, round_to_even: bool=True):
    mask = tir.const((1 << 16) - 1, "uint32")
    res = []
    for data in [v0, v1]:
        u32_val = tir.reinterpret("uint32", data)
        if round_to_even:
            rounding_bias = ((u32_val >> tir.const(16, "uint32")) & tir.const(1, "uint32")) + tir.const(0x7FFF, "uint32")
            u32_val += rounding_bias
        res.append((u32_val >> tir.const(16, "uint32")) & mask)
    return res[0] | (res[1] << tir.const(16, "uint32"))


def _tir_u32_to_bf16x2_to_f32x2(x: tir.PrimExpr):
    mask = tir.const((1 << 16) - 1, "uint32")
    x0 = x & mask
    x1 = (x >> 16) & mask
    return (tir.reinterpret("float32", x << tir.const(16, "uint32")) for x in [x0, x1])


def _tir_u32_to_int_to_float(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    assert val.dtype == "uint32"
    mask = tvm.tir.const((1 << nbit) - 1, "uint32")
    return tir.Cast(dtype, (val >> (pos * nbit).astype("uint32")) & mask)


def _tir_packed_uint_to_uint_to_float(storage_nbit: int):
    storage_dtype = "uint" + str(storage_nbit)

    def f_convert(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        assert val.dtype == storage_dtype
        max_int_value = (1 << (nbit - 1)) - 1
        return ((val >> (pos.astype("uint32") * tir.const(nbit, "uint32"))) & tir.const((1 << nbit) - 1, "uint32")).astype(dtype) - tir.const(max_int_value, dtype)

    return f_convert


def _tir_packed_int_to_int_to_float(storage_nbit: int):
    storage_dtype = "int" + str(storage_nbit)

    def f_convert(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        assert val.dtype == storage_dtype
        mask = tir.const((1 << nbit) - 1, "int32")
        unextended = (val >> (pos.astype("int32") * tir.const(nbit, "int32"))) & mask
        return tir.Cast(dtype, (unextended << tir.const(32 - nbit, "int32")) >> tir.const(32 - nbit, "int32"))

    return f_convert


def _tir_f32_to_uint_to_f4(val: tir.PrimExpr):
    assert val.dtype == "float32"
    val_u32 = tir.reinterpret("uint32", val)
    # e_f32 >  120 -> e_f4 = min(e_f32 - 120 + M_h, 7)
    # e_f32 == 120 -> e_f4 = 1
    # e_f32 < 120 -> e_f4 = 0
    m_h = (val_u32 >> tir.const(22, "uint32")) & tir.const(1, "uint32")
    e_f32 = (val_u32 >> tir.const(23, "uint32")) & tir.const(255, "uint32")
    s = (val_u32 >> tir.const(31, "uint32"))
    e_f4 = tir.Select(e_f32 > tir.const(120, "uint32"), tir.Min(e_f32 - tir.const(120, "uint32") + m_h, tir.const(7, "uint32")), tir.Select(e_f32 == tir.const(120, "uint32"), tir.const(1, "uint32"), tir.const(0, "uint32")))
    return (s << tir.const(3, "uint32")) | e_f4


def _tir_f16_to_uint_to_f4(val: tir.PrimExpr):
    assert val.dtype == "float16"
    val_u32 = tir.Cast("uint32", tir.reinterpret("uint16", val))
    m_h = (val_u32 >> tir.const(9, "uint32")) & tir.const(1, "uint32")
    e_f16 = (val_u32 >> tir.const(10, "uint32")) & tir.const(31, "uint32")
    s = (val_u32 >> tir.const(15, "uint32"))
    e_f4 = tir.Select(e_f16 > tir.const(8, "uint32"), tir.Min(e_f16 - tir.const(8, "uint32") + m_h, tir.const(7, "uint32")), tir.Select(e_f16 == tir.const(8, "uint32"), tir.const(1, "uint32"), tir.const(0, "uint32")))
    return (s << tir.const(3, "uint32")) | e_f4


def _tir_u32_to_f4_to_f32(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    assert nbit == 4
    assert dtype == "float32"
    assert val.dtype == "uint32"
    # e_f4 == 0 -> e_f32 = 0
    # e_f4 != 0 -> e_f32 = e_f4 + 120 = e_f4 | (1111000)_2
    mask = tvm.tir.const((1 << nbit) - 1, "uint32")
    f4 = (val >> (pos.astype("uint32") * tir.const(nbit, "uint32"))) & mask
    s = f4 >> tir.const(3, "uint32")
    e_f4 = f4 & tir.const(7, "uint32")
    e_f32 = e_f4 | tir.const(120, "uint32")
    val_f32 = tir.reinterpret("float32", (e_f32 | (s << tir.const(8, "uint32"))) << tir.const(23, "uint32"))
    return tir.Select(e_f4 == tir.const(0, "uint32"), tir.const(0, "float32"), val_f32)


def _tir_u32_to_f4_to_f16(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    assert nbit == 4
    assert dtype == "float16"
    assert val.dtype == "uint32"
    # e_f4 == 0 -> e_f16 = 0
    # e_f4 != 0 -> e_f16 = e_f4 + 8 = e_f4 | (1000)_2
    mask = tvm.tir.const((1 << nbit) - 1, "uint32")
    f4 = (val >> (pos.astype("uint32") * tir.const(nbit, "uint32"))) & mask
    s = f4 >> tir.const(3, "uint32")
    e_f4 = f4 & tir.const(7, "uint32")
    e_f16 = e_f4 | tir.const(8, "uint32")
    val_f16 = tir.reinterpret("float16", (e_f16 | (s << tir.const(5, "uint32"))) << tir.const(10, "uint32"))
    return tir.Select(e_f4 == tir.const(0, "uint32"), tir.const(0, "float16"), val_f16)

def decoding_func(
    sym: bool,
    group_size: int,
    nbit: int,
    mode: str,
    storage_nbit: int,
    dim_length: tir.PrimExpr,
    data_transposed: bool=True,
    transpose_output: bool=False,
    dtype: str = "float32"
):
    def te_decode_asym(*args):
        n_float_per_u32 = 32 // nbit
        data = args[0]
        if dtype == "float32":
            scale_bias_bf16x2 = args[1]
        else:
            scale, min_value = args[1], args[2]

        def f_decode_asym(i, j):
            if data_transposed:
                data_float = _tir_u32_to_int_to_float(nbit, data[i // n_float_per_u32, j], i % n_float_per_u32, dtype=dtype)
                if dtype == "float32":
                    scale_float, bias_float = _tir_u32_to_bf16x2_to_f32x2(scale_bias_bf16x2[i // group_size, j])
                else:
                    scale_float, bias_float = scale[i // group_size, j], min_value[i // group_size, j]
            else:
                data_float = _tir_u32_to_int_to_float(nbit, data[i, j // n_float_per_u32], j % n_float_per_u32, dtype=dtype)
                if dtype == "float32":
                    scale_float, bias_float = _tir_u32_to_bf16x2_to_f32x2(scale_bias_bf16x2[i, j // group_size])
                else:
                    scale_float, bias_float = scale[i, j // group_size], min_value[i, j // group_size]
            w = data_float * scale_float + bias_float
            return w

        shape = (dim_length, data.shape[1]) if data_transposed else (data.shape[0], dim_length)
        w = te.compute(shape=shape, fcompute=f_decode_asym, name="decode")
        if transpose_output:
            w = topi.transpose(w)
        return w

    def te_decode_sym(data, scale):
        n_float_per_int = storage_nbit // nbit

        def f_decode_sym(i, j):
            f_convert = _tir_packed_uint_to_uint_to_float(storage_nbit) if mode.startswith("int") else (_tir_u32_to_f4_to_f32 if dtype == "float32" else _tir_u32_to_f4_to_f16)
            if data_transposed:
                data_float = f_convert(nbit, data[i // n_float_per_int, j], i % n_float_per_int, dtype=dtype)
                scale_float = scale[i // group_size, j]
            else:
                data_float = f_convert(nbit, data[i, j // n_float_per_int], j % n_float_per_int, dtype=dtype)
                scale_float = scale[i, j // group_size]
            return data_float * scale_float

        shape = (dim_length, data.shape[1]) if data_transposed else (data.shape[0], dim_length)
        w = te.compute(shape=shape, fcompute=f_decode_sym, name="decode")
        if transpose_output:
            w = topi.transpose(w)
        return w

    return te_decode_sym if sym else te_decode_asym


def encoding_func(
    sym: bool,
    group_size: int,
    nbit: int,
    mode: str,
    storage_nbit: int,
    transpose: bool=True,
    dtype: str = "float32"
):
    def te_encode_asym(weight: te.Tensor):
        assert weight.shape[1] % group_size == 0
        n_group = weight.shape[1] // group_size
        n_float_per_u32 = 32 // nbit

        scale_min_shape = (weight.shape[0], n_group)
        k = te.reduce_axis((0, group_size), name="k")
        min_value = te.compute(shape=scale_min_shape, fcompute=lambda i, j: te.min(weight[i, j * group_size + k], axis=k), name="min_value")
        max_value = te.compute(shape=scale_min_shape, fcompute=lambda i, j: te.max(weight[i, j * group_size + k], axis=k), name="max_value")
        scale = te.compute(shape=scale_min_shape, fcompute=lambda i, j: (max_value[i, j] - min_value[i, j]) / tir.const((1 << nbit) - 1, dtype), name="scale")

        def f_scale_weight(i, j):
            group_idx = j // group_size
            w_scaled = tir.round((weight[i, j] - min_value[i, group_idx]) / scale[i, group_idx]).astype("int32")
            w_scaled = T.min(T.max(w_scaled, tir.const(0, "int32")), tir.const((1 << nbit) - 1, "int32"))
            w_scaled = w_scaled.astype("uint32")
            return w_scaled

        k = te.reduce_axis((0, n_float_per_u32), name="k")
        reducer = te.comm_reducer(fcombine=lambda x, y: tir.bitwise_or(x, y), fidentity=lambda dtype: tir.const(0, dtype), name="bitwise_or")
        if dtype == "float32":
            if transpose:
                w_gathered = te.compute(shape=(weight.shape[1] // n_float_per_u32, weight.shape[0]), fcompute=lambda j, i: reducer(f_scale_weight(i, j * n_float_per_u32 + k) << (k * nbit).astype("uint32"), axis=k), name="w_gathered")
                scale_bias = te.compute(shape=(n_group, weight.shape[0]), fcompute=lambda j, i: _tir_f32x2_to_bf16x2_to_u32(scale[i, j], min_value[i, j], round_to_even=True), name="scale_min")
            else:
                w_gathered = te.compute(shape=(weight.shape[0], weight.shape[1] // n_float_per_u32), fcompute=lambda i, j: reducer(f_scale_weight(i, j * n_float_per_u32 + k) << (k * nbit).astype("uint32"), axis=k), name="w_gathered")
                scale_bias = te.compute(shape=(weight.shape[0], n_group), fcompute=lambda i, j: _tir_f32x2_to_bf16x2_to_u32(scale[i, j], min_value[i, j], round_to_even=True), name="scale_min")
            return w_gathered, scale_bias
        else:
            if transpose:
                w_gathered = te.compute(shape=(weight.shape[1] // n_float_per_u32, weight.shape[0]), fcompute=lambda j, i: reducer(f_scale_weight(i, j * n_float_per_u32 + k) << (k * nbit).astype("uint32"), axis=k), name="w_gathered")
                scale = te.compute(shape=(n_group, weight.shape[0]), fcompute=lambda j, i: scale[i, j], name="scale_transpose")
                min_value = te.compute(shape=(n_group, weight.shape[0]), fcompute=lambda j, i: min_value[i, j], name="min_transpose")
            else:
                w_gathered = te.compute(shape=(weight.shape[0], weight.shape[1] // n_float_per_u32), fcompute=lambda i, j: reducer(f_scale_weight(i, j * n_float_per_u32 + k) << (k * nbit).astype("uint32"), axis=k), name="w_gathered")
            return w_gathered, scale, min_value

    def te_encode_sym(weight: te.Tensor):
        n_group = tir.ceildiv(weight.shape[1], group_size)
        n_float_per_int = storage_nbit // nbit
        max_int_value = (1 << (nbit - 1)) - 1
        assert group_size % n_float_per_int == 0

        scale_min_shape = (weight.shape[0], n_group)
        k = te.reduce_axis((0, group_size), name="k")
        max_abs_value = te.compute(shape=scale_min_shape, fcompute=lambda i, j: te.max(tir.if_then_else(j * group_size + k < weight.shape[1], te.abs(weight[i, j * group_size + k]), tir.min_value(dtype)), axis=k), name="max_abs_value")

        def f_compute_scale(i, j):
            max_value = tir.max(max_abs_value[i, j], tir.const(1e-4, dtype))
            return (max_value / tir.const(max_int_value, dtype)) if mode.startswith("int") else max_value

        scale = te.compute(shape=scale_min_shape, fcompute=f_compute_scale, name="scale")
        storage_dtype = ("uint" + str(storage_nbit)) if mode.startswith("int") else "uint32"

        def f_scale_weight(i, j):
            group_idx = j // group_size
            if mode.startswith("int"):
                w_scaled = tir.round(weight[i, j] / scale[i, group_idx] + tir.const(max_int_value, dtype))
                w_scaled = T.min(T.max(w_scaled, tir.const(0, dtype)), tir.const(max_int_value * 2, dtype)).astype(storage_dtype)
                return w_scaled
            else:
                f_convert = _tir_f32_to_uint_to_f4 if dtype == "float32" else _tir_f16_to_uint_to_f4
                return f_convert(weight[i, j] / scale[i, group_idx])

        k = te.reduce_axis((0, n_float_per_int), name="k")
        reducer = te.comm_reducer(fcombine=lambda x, y: tir.bitwise_or(x, y), fidentity=lambda dtype: tir.const(0, dtype), name="bitwise_or")
        n_i32 = tir.ceildiv(group_size, n_float_per_int) * n_group
        if transpose:
            w_gathered = te.compute(shape=(n_i32, weight.shape[0]), fcompute=lambda j, i: reducer(tir.if_then_else(j * n_float_per_int + k < weight.shape[1], f_scale_weight(i, j * n_float_per_int + k) << (k.astype(storage_dtype) * tir.const(nbit, storage_dtype)), tir.const(0, storage_dtype)), axis=k), name="w_gathered")
            scale = te.compute(shape=(n_group, weight.shape[0]), fcompute=lambda j, i: scale[i, j])
        else:
            w_gathered = te.compute(shape=(weight.shape[0], n_i32), fcompute=lambda i, j: reducer(tir.if_then_else(j * n_float_per_int + k < weight.shape[1], f_scale_weight(i, j * n_float_per_int + k) << (k.astype(storage_dtype) * tir.const(nbit, storage_dtype)), tir.const(0, storage_dtype)), axis=k), name="w_gathered")
        return w_gathered, scale

    return te_encode_sym if sym else te_encode_asym
# fmt: on


@dataclass
class GroupQuantizationSpec:
    """The quantization specification for group quantization algorithm."""

    mode: Literal["int3", "int4"]
    sym: bool
    storage_nbit: int
    group_size: int
    transpose: bool
    dtype: str

    def get_quantize_func(self, param_info: relax.TensorStructInfo=None):
        return encoding_func(
            sym=self.sym,
            group_size=self.group_size,
            nbit=int(self.mode[-1]),
            mode=self.mode,
            storage_nbit=self.storage_nbit,
            transpose=self.transpose,
            dtype=self.dtype,
        )

    def get_dequantize_func(
        self,
        param_info: relax.TensorStructInfo,
    ):
        return decoding_func(
            sym=self.sym,
            group_size=self.group_size,
            nbit=int(self.mode[-1]),
            mode=self.mode,
            storage_nbit=self.storage_nbit,
            dim_length=param_info.shape.values[-1],
            data_transposed=self.transpose,
            transpose_output=self.transpose,
            dtype=self.dtype,
        )

    def get_dequantize_sinfo(self, param_info: relax.TensorStructInfo):
        return [
            relax.TensorStructInfo(
                shape=[
                    param_info.shape[0],
                    param_info.shape[1] * int(self.mode[-1]) // self.storage_nbit,
                ],
                dtype="uint" + str(self.storage_nbit),
            ),
            relax.TensorStructInfo(
                shape=[param_info.shape[0], param_info.shape[1] // self.group_size],
                dtype=self.dtype,
            ),
        ]


q4f16_1 = GroupQuantizationSpec(
    dtype="float16",
    mode="int4",
    sym=True,
    storage_nbit=32,
    group_size=32,
    transpose=False,
)


q3f16_1 = GroupQuantizationSpec(
    dtype="float16",
    mode="int3",
    sym=True,
    storage_nbit=16,
    group_size=40,
    transpose=False,
)


if __name__ == "__main__":
    quantize_mode = q4f16_1
    param = relax.Var("param", relax.TensorStructInfo([4096, 11008], "float16"))
    quantized_sinfo = quantize_mode.get_dequantize_sinfo(param.struct_info)
    quantized = [relax.Var("weight", quantized_sinfo[0]), relax.Var("scale", quantized_sinfo[1])]
    bb = relax.BlockBuilder()
    with bb.function("main", [*quantized]):
        with bb.dataflow():
            w = bb.emit_te(
                quantize_mode.get_dequantize_func(param.struct_info),
                *quantized,
                primfunc_name_hint="dequantize",
            )
            out = bb.emit_output(w)
        bb.emit_func_output(out)
    mod = bb.get()
    mod.show()

    bb = relax.BlockBuilder()
    with bb.function("main", [param]):
        with bb.dataflow():
            w_q = bb.emit_te(
                quantize_mode.get_quantize_func(),
                param,
                primfunc_name_hint="quantize",
            )
            out = bb.emit_output(w)
        bb.emit_func_output(out)
    mod = bb.get()
    mod.show()
