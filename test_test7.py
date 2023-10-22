from tvm.script import tir as T, ir as I
import tvm

@I.ir_module
class Module:
    I.module_attrs({"runtime": None})
    @T.prim_func
    def default_function(lv2608: T.Buffer((T.int64(11008), T.int64(512)), "uint32"), lv2609: T.Buffer((T.int64(11008), T.int64(128)), "float16"), p_lv7330: T.handle, p_output0: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_80", "host": {"keys": ["cpu"], "kind": "llvm", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "l2_cache_size_bytes": 41943040, "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "registers_per_block": 65536, "tag": "", "thread_warp_size": 32}), "tir.is_entry_func": T.bool(True), "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        b = T.int64()
        lv7330 = T.match_buffer(p_lv7330, (b, T.int64(512), T.int64(4096)), "float16")
        p_output0_intermediate_1 = T.allocate([11272192], "float16x4", "global")
        blockIdx_x = T.launch_thread("blockIdx.x", b * T.int64(344))
        lv7330_reindex_shared_dyn = T.allocate([4096], "float16", "shared.dyn")
        threadIdx_z = T.launch_thread("threadIdx.z", T.int64(2))
        threadIdx_y = T.launch_thread("threadIdx.y", T.int64(2))
        threadIdx_x = T.launch_thread("threadIdx.x", T.int64(32))
        for ax3_0_0 in range(T.int64(128)):
            T.tvm_storage_sync("shared.dyn")
            for ax0_ax1_fused_0 in range(T.int64(8)):
                lv7330_reindex_shared_dyn_1 = T.Buffer((T.int64(4096),), "float16", data=lv7330_reindex_shared_dyn, scope="shared.dyn")
                lv7330_1 = T.Buffer((b * T.int64(2097152),), "float16", data=lv7330.data)
                lv7330_reindex_shared_dyn_1[ax0_ax1_fused_0 * T.int64(512) + threadIdx_z * T.int64(256) + threadIdx_y * T.int64(128) + threadIdx_x // T.int64(8) * T.int64(32) + T.bitwise_xor(threadIdx_x % T.int64(8) // T.int64(2), threadIdx_y * T.int64(2) + threadIdx_x // T.int64(16)) * T.int64(8) + threadIdx_x % T.int64(2) * T.int64(4):ax0_ax1_fused_0 * T.int64(512) + threadIdx_z * T.int64(256) + threadIdx_y * T.int64(128) + threadIdx_x // T.int64(8) * T.int64(32) + T.bitwise_xor(threadIdx_x % T.int64(8) // T.int64(2), threadIdx_y * T.int64(2) + threadIdx_x // T.int64(16)) * T.int64(8) + threadIdx_x % T.int64(2) * T.int64(4) + T.int64(4)] = lv7330_1[blockIdx_x // T.int64(86) * T.int64(524288) + ax0_ax1_fused_0 * T.int64(65536) + threadIdx_z * T.int64(32768) + threadIdx_y * T.int64(16384) + threadIdx_x // T.int64(8) * T.int64(4096) + ax3_0_0 * T.int64(32) + threadIdx_x % T.int64(8) * T.int64(4):blockIdx_x // T.int64(86) * T.int64(524288) + ax0_ax1_fused_0 * T.int64(65536) + threadIdx_z * T.int64(32768) + threadIdx_y * T.int64(16384) + threadIdx_x // T.int64(8) * T.int64(4096) + ax3_0_0 * T.int64(32) + threadIdx_x % T.int64(8) * T.int64(4) + T.int64(4)]
Module.show(None, False)
