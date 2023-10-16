from tvm.script import tir as T, ir as I
import tvm

@I.ir_module
class Module:
    @T.prim_func
    def default_function():
        blockIdx_x = T.launch_thread("blockIdx.x", 1024)
        threadIdx_x = T.launch_thread("threadIdx.x", 32)
        A_reindex_shared_dyn = T.allocate([16384], "float16", "shared.dyn")
        for ax3_0_1 in range(2):
            A_reindex_shared_dyn_warp = T.allocate([1024], "float16", "warp")
            T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", A_reindex_shared_dyn_warp, threadIdx_x * 8, T.tvm_access_ptr(T.type_annotation("float16"), A_reindex_shared_dyn, ax3_0_1 * 16, T.int64(512), 1), threadIdx_x % 16 * 32 + threadIdx_x // 16 * 8)
        for ax3_0_1 in range(2):
            A_reindex_shared_dyn_warp = T.allocate([1024], "float16", "warp")
            T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", A_reindex_shared_dyn_warp, threadIdx_x * 8, T.tvm_access_ptr(T.type_annotation("float16"), A_reindex_shared_dyn, ax3_0_1 * 16 + 4096, T.int64(512), 1), threadIdx_x % 16 * 32 + threadIdx_x // 16 * 8)


target, dev = tvm.target.Target("nvidia/nvidia-a100"), tvm.cuda()

with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
    ex = tvm.build(Module["default_function"], target=target)
