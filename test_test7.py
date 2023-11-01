from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T, ir as I, relax as R
import tvm
from tvm import tir

a = tir.SizeVar("a", "int64")
analyzer = Analyzer()
print(analyzer.rewrite_simplify(tir.Max(tir.const(0, "int64"), a)))
