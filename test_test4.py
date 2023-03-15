from typing import Callable, List, Tuple
# import numpy as np
# import numpy.random
# import pytest
import tvm
# from tvm._ffi.base import TVMError
# from tvm.arith.analyzer import Analyzer
import tvm.script
import tvm.testing
from tvm import relax
from tvm import relax as rx
from tvm import te, tir
from tvm.ir.base import assert_structural_equal
from tvm.relax import Function, Var
from tvm.relax.block_builder import BlockBuilder
from tvm.relax.expr import Expr
from tvm import topi, relay
from tvm.relax.op import add, divide, multiply, sqrt, subtract
from tvm.relax.struct_info import TensorStructInfo
from tvm.relax.transform import LegalizeOps
from tvm.runtime.container import tuple_object
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T
from tvm.tir.function import PrimFunc
from tvm.relax import training
from tvm.relax.frontend.torch import from_fx
class cl:
    def a(self):
        print("a")
    def b(self):
        print("b")

class cl1:
    def __init__(self, val):
        self.val = val


    def b(self):
        print("c")

    def __getattr__(self, name):
        return getattr(self.val, name)


a = cl()
b = cl1(a)
b.b()
