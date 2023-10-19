import tvm
from tvm import IRModule, relax, tir
from tvm.relax.dpl.pattern import GlobalVarPattern, TuplePattern, is_op, wildcard


def check_decoding(ctx: relax.transform.PatternCheckContext) -> bool:
    call = ctx.annotated_expr["w"]
    if not isinstance(call, relax.Call):
        return False
    gv = call.args[0]
    if not isinstance(gv, relax.GlobalVar):
        return False
    return "dequantize" in gv.name_hint


def check_matmul(ctx: relax.transform.PatternCheckContext) -> bool:
    call = ctx.annotated_expr["matmul"]
    if not isinstance(call, relax.Call):
        return False
    gv = call.args[0]
    if not isinstance(gv, relax.GlobalVar):
        return False
    return "matmul" in gv.name_hint


def decode_matmul_pattern():
    w_scaled = wildcard()
    aux_tensors = [wildcard()]
    x = wildcard()
    w = is_op("relax.call_tir")(
        GlobalVarPattern(),
        TuplePattern([w_scaled, *aux_tensors]),
        add_constraint=False,
    )
    matmul_args = [w, x]
    matmul = is_op("relax.call_tir")(
        GlobalVarPattern(), TuplePattern(matmul_args), add_constraint=False
    )

    annotations = {
        "matmul": matmul,
        "w": w,
        "x": x,
        "w_scaled": w_scaled,
    }

    def f_pattern_check(ctx: relax.transform.PatternCheckContext) -> bool:
        return check_decoding(ctx) and check_matmul(ctx)

    return matmul, annotations, f_pattern_check
