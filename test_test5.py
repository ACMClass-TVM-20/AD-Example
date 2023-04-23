import tvm
from tvm import relax
from tvm._ffi.registry import get_global_func
from tvm.meta_schedule.runner.runner import Runner
from tvm.meta_schedule.runner.utils import alloc_argument_common
from tvm.runtime.ndarray import NDArray
from tvm import te, topi, meta_schedule as ms
import numpy as np
from tvm.script import tir as T, ir as I
from tvm.tir import Schedule
# @tvm.script.ir_module
# class InputModule:
#     @R.function
#     def main(
#         x: R.Tensor((1, 1, 16, 16), "float32"), y: R.Tensor((1, 1, 3, 3), "float32")
#     ):
#         with R.dataflow():
#             z1 = R.full((3, 3), )
#             R.output(z1)
#         return z1

# target, dev = tvm.target.Target("nvidia/geforce-rtx-3080"), tvm.cuda()

# mod = relax.transform.LegalizeOps()(InputModule)
# mod.show(None, False)
# with target:
#     mod = tvm.tir.transform.DefaultGPUSchedule()(mod)
# mod.show(None, False)


@tvm.script.ir_module
class mod:
    @T.prim_func
    def main(T_full: T.Buffer((), "float32")):
        # with T.block("root"):
        with T.block("T_full"):
            vi = T.axis.spatial(1, 0)
            T.reads()
            T.writes(T_full[()])
            T_full[()] = T.float32(1)
sch = Schedule(mod)

sch = ms.schedule_rule.InlineConstantScalars().apply(sch, sch.get_block("T_full"))[0]
sch.mod.show()


# ex = relax.build(mod, target)
# vm = relax.VirtualMachine(ex, dev)
# inputs = [np.zeros((1, 1, 16, 16), "float32"), np.zeros((1, 1, 3, 3), "float32")]
# res = vm["main"](tvm.nd.array(inputs[0], dev), tvm.nd.array(inputs[1], dev))
# print(res)

# work_dir = "/home/yxdong/relax-mlcai/other-repos/AD-Example/tmp/tune1"
# def random_fill(data: NDArray):
#     random_fill_for_measure = get_global_func("tvm.contrib.random.random_fill_for_measure")
#     if 'int' in data.dtype:
#         new_data = np.zeros(data.shape, dtype=data.dtype)
#         data.copyfrom(new_data)
#     else:
#         random_fill_for_measure(data)

# def alloc_argument(device, args_info, alloc_repeat):
#     return alloc_argument_common(random_fill, device, args_info, alloc_repeat)

# runner = Runner.create("local", f_alloc_argument=alloc_argument)

# tune_relax(mod, {}, target, work_dir, 80, runner=runner)
# # tune_relax(mod, {}, target, work_dir, 80)
# # with tempfile.TemporaryDirectory() as work_dir:
# with target, tvm.transform.PassContext(trace=Trace(mod)):
#     mod_new = tvm.transform.Sequential(
#         [
#             # relax.transform.MetaScheduleTuneIRMod(
#             #     params={}, work_dir=work_dir, max_trials_global=80
#             # ),
#             relax.transform.MetaScheduleApplyDatabase(work_dir),
#         ]
#     )(mod)
# assert relax.analysis.well_formed(mod_new)
# mod_new.without_attr("external_mods").show()
