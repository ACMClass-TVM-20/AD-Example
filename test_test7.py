from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T, ir as I, relax as R
import tvm
from tvm import tir, ir

b = tir.Var("b", "int64")
v = tir.Var("v", "int64")
analyzer = Analyzer()
analyzer.bind(v, ir.Range(0, b * 128))
print(analyzer.int_set(v, None))
tmp = analyzer.rewrite_simplify(v // 128 < b)
print(type(tmp))
print(tmp)
