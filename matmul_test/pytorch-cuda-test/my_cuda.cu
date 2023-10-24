#include <cuda_fp16.h>
#include <torch/extension.h>

__device__ half max(half a, half b) { return __hgt(__half(a), __half(b)) ? a : b; }
__device__ half min(half a, half b) { return __hlt(__half(a), __half(b)) ? a : b; }

// Pack two half values.
static inline __device__ __host__ unsigned __pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short*)&x);
  unsigned v1 = *((unsigned short*)&y);
  return (v1 << 16) | v0;
}

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
  static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) { \
    float tmp_x = __half2float(x);                                        \
    float tmp_y = __half2float(y);                                        \
    float result = FP32_MATH_NAME(tmp_x, tmp_y);                          \
    return __float2half(result);                                          \
  }

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
  static inline __device__ __host__ half HALF_MATH_NAME(half x) {        \
    float tmp_x = __half2float(x);                                       \
    float result = FP32_MATH_NAME(tmp_x);                                \
    return __float2half(result);                                         \
  }

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY
__forceinline__ __device__ unsigned int cast_smem_ptr_to_int(const void* const smem_ptr) {
  unsigned int smem_int;
  asm volatile("{ .reg .u64 smem_int; cvta.to.shared.u64 smem_int, %1; cvt.u32.u64 %0, smem_int; }"
               : "=r"(smem_int)
               : "l"(smem_ptr));
  return smem_int;
}

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long
extern "C" __global__ void __launch_bounds__(128)
    fused_fused_decode3_fused_NT_matmul21_cast21_add5_silu2_kernel(
        uint* __restrict__ lv2608, half* __restrict__ lv2609, half* __restrict__ lv7330,
        half* __restrict__ p_output0_intermediate, int64_t b) {
  extern __shared__ uchar buf_dyn_shmem[];
  float var_NT_matmul_intermediate_reindex_shared_dyn_warp[128];
  half lv7330_reindex_shared_dyn_warp[32];
  half p_output0_intermediate_1_reindex_shared_dyn_warp[32];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 4; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 4; ++ax2_0_3_init) {
      for (int i = 0; i < 8; ++i) {
        var_NT_matmul_intermediate_reindex_shared_dyn_warp[((ax1_0_3_init * 32) +
                                                            (ax2_0_3_init * 8)) +
                                                           i] = 0.0;
      };
    }
  }
  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)128; ++ax3_0_0) {
    __syncthreads();
    for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)4; ++ax0_ax1_fused_0) {
      *(uint4*)(((half*)buf_dyn_shmem) +
                ((((((ax0_ax1_fused_0 * (int64_t)1024) + (((int64_t)threadIdx.z) * (int64_t)512)) +
                    (((int64_t)threadIdx.y) * (int64_t)256)) +
                   ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)32)) +
                  (((((int64_t)threadIdx.x) & (int64_t)3) ^
                    (((int64_t)threadIdx.x) >> (int64_t)3)) *
                   (int64_t)8)) +
                 (int64_t)4096)) =
          *(uint4*)(lv7330 + ((((((((((int64_t)blockIdx.x) / (int64_t)86) * (int64_t)524288) +
                                   (ax0_ax1_fused_0 * (int64_t)131072)) +
                                  (((int64_t)threadIdx.z) * (int64_t)65536)) +
                                 (((int64_t)threadIdx.y) * (int64_t)32768)) +
                                ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)4096)) +
                               (ax3_0_0 * (int64_t)32)) +
                              ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)));
    }
    *(uint4*)(((uint*)buf_dyn_shmem) + ((((((int64_t)threadIdx.z) * (int64_t)256) +
                                          (((int64_t)threadIdx.y) * (int64_t)128)) +
                                         (((int64_t)threadIdx.x) * (int64_t)4)) +
                                        (int64_t)4096)) =
        *(uint4*)(lv2608 + ((((((((int64_t)blockIdx.x) % (int64_t)86) * (int64_t)65536) +
                               (((int64_t)threadIdx.z) * (int64_t)32768)) +
                              (((int64_t)threadIdx.y) * (int64_t)16384)) +
                             (((int64_t)threadIdx.x) * (int64_t)512)) +
                            (ax3_0_0 * (int64_t)4)));
    ((half*)buf_dyn_shmem)[(
        (((((int64_t)threadIdx.z) * (int64_t)64) + (((int64_t)threadIdx.y) * (int64_t)32)) +
         ((int64_t)threadIdx.x)) +
        (int64_t)9216)] = lv2609[((((((((int64_t)blockIdx.x) % (int64_t)86) * (int64_t)16384) +
                                     (((int64_t)threadIdx.z) * (int64_t)8192)) +
                                    (((int64_t)threadIdx.y) * (int64_t)4096)) +
                                   (((int64_t)threadIdx.x) * (int64_t)128)) +
                                  ax3_0_0)];
    for (int64_t ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < (int64_t)32; ++ax0_ax1_fused_0_1) {
      __syncthreads();
      ((half*)buf_dyn_shmem)[(
          ((((ax0_ax1_fused_0_1 * (int64_t)128) + (((int64_t)threadIdx.z) * (int64_t)64)) +
            (((int64_t)threadIdx.y) * (int64_t)32)) +
           (((((int64_t)threadIdx.x) >> (int64_t)3) ^
             (((ax0_ax1_fused_0_1 & (int64_t)1) * (int64_t)2) + ((int64_t)threadIdx.z))) *
            (int64_t)8)) +
          (((int64_t)threadIdx.x) & (int64_t)7))] =
          ((((half)((((uint*)buf_dyn_shmem)[(((((ax0_ax1_fused_0_1 * (int64_t)16) +
                                                (((int64_t)threadIdx.z) * (int64_t)8)) +
                                               (((int64_t)threadIdx.y) * (int64_t)4)) +
                                              (((int64_t)threadIdx.x) >> (int64_t)3)) +
                                             (int64_t)4096)] >>
                     (((uint)(((int64_t)threadIdx.x) & (int64_t)7)) * (uint)4)) &
                    (uint)15)) -
            __float2half_rn(7.000000e+00f)) *
           ((half*)buf_dyn_shmem)[(
               (((ax0_ax1_fused_0_1 * (int64_t)4) + (((int64_t)threadIdx.z) * (int64_t)2)) +
                ((int64_t)threadIdx.y)) +
               (int64_t)9216)]);
    }
    __syncthreads();
    for (int64_t ax3_0_1 = 0; ax3_0_1 < (int64_t)2; ++ax3_0_1) {
      for (int64_t ax0_0 = 0; ax0_0 < (int64_t)4; ++ax0_0) {
        {
          unsigned int addr = cast_smem_ptr_to_int(
              (&(((half*)buf_dyn_shmem)[(
                  ((((((int64_t)threadIdx.z) * (int64_t)2048) + (ax0_0 * (int64_t)512)) +
                    ((((int64_t)threadIdx.x) & (int64_t)15) * (int64_t)32)) +
                   ((((ax3_0_1 * (int64_t)2) + (((int64_t)threadIdx.x) >> (int64_t)4)) ^
                     ((((int64_t)threadIdx.x) & (int64_t)7) >> (int64_t)1)) *
                    (int64_t)8)) +
                  (int64_t)4096)])) +
              (int64_t)0);
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
              "{%0, %1, %2, %3}, [%4];\n"
              : "=r"(((unsigned*)(lv7330_reindex_shared_dyn_warp + (ax0_0 * (int64_t)8)))[0]),
                "=r"(((unsigned*)(lv7330_reindex_shared_dyn_warp + (ax0_0 * (int64_t)8)))[1]),
                "=r"(((unsigned*)(lv7330_reindex_shared_dyn_warp + (ax0_0 * (int64_t)8)))[2]),
                "=r"(((unsigned*)(lv7330_reindex_shared_dyn_warp + (ax0_0 * (int64_t)8)))[3])
              : "r"(addr));
        }
      }
      for (int64_t ax0_0_1 = 0; ax0_0_1 < (int64_t)4; ++ax0_0_1) {
        {
          unsigned int addr = cast_smem_ptr_to_int(
              (&(((half*)buf_dyn_shmem)[(
                  ((((((int64_t)threadIdx.y) * (int64_t)2048) + (ax0_0_1 * (int64_t)512)) +
                    ((((int64_t)threadIdx.x) >> (int64_t)4) * (int64_t)256)) +
                   ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)32)) +
                  ((((ax3_0_1 * (int64_t)2) +
                     ((((int64_t)threadIdx.x) & (int64_t)15) >> (int64_t)3)) ^
                    ((((int64_t)threadIdx.x) & (int64_t)7) >> (int64_t)1)) *
                   (int64_t)8))])) +
              (int64_t)0);
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
              "{%0, %1, %2, %3}, [%4];\n"
              : "=r"(((unsigned*)(p_output0_intermediate_1_reindex_shared_dyn_warp +
                                  (ax0_0_1 * (int64_t)8)))[0]),
                "=r"(((unsigned*)(p_output0_intermediate_1_reindex_shared_dyn_warp +
                                  (ax0_0_1 * (int64_t)8)))[1]),
                "=r"(((unsigned*)(p_output0_intermediate_1_reindex_shared_dyn_warp +
                                  (ax0_0_1 * (int64_t)8)))[2]),
                "=r"(((unsigned*)(p_output0_intermediate_1_reindex_shared_dyn_warp +
                                  (ax0_0_1 * (int64_t)8)))[3])
              : "r"(addr));
        }
      }
      for (int ax1_0_3 = 0; ax1_0_3 < 4; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 4; ++ax2_0_3) {
          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(((float*)(var_NT_matmul_intermediate_reindex_shared_dyn_warp +
                                 ((((int64_t)ax1_0_3) * (int64_t)32) +
                                  (((int64_t)ax2_0_3) * (int64_t)8))))[0]),
                  "=f"(((float*)(var_NT_matmul_intermediate_reindex_shared_dyn_warp +
                                 ((((int64_t)ax1_0_3) * (int64_t)32) +
                                  (((int64_t)ax2_0_3) * (int64_t)8))))[1]),
                  "=f"(((float*)(var_NT_matmul_intermediate_reindex_shared_dyn_warp +
                                 ((((int64_t)ax1_0_3) * (int64_t)32) +
                                  (((int64_t)ax2_0_3) * (int64_t)8))))[2]),
                  "=f"(((float*)(var_NT_matmul_intermediate_reindex_shared_dyn_warp +
                                 ((((int64_t)ax1_0_3) * (int64_t)32) +
                                  (((int64_t)ax2_0_3) * (int64_t)8))))[3])
                : "r"(((unsigned*)(lv7330_reindex_shared_dyn_warp +
                                   (((int64_t)ax1_0_3) * (int64_t)8)))[0]),
                  "r"(((unsigned*)(lv7330_reindex_shared_dyn_warp +
                                   (((int64_t)ax1_0_3) * (int64_t)8)))[1]),
                  "r"(((unsigned*)(lv7330_reindex_shared_dyn_warp +
                                   (((int64_t)ax1_0_3) * (int64_t)8)))[2]),
                  "r"(((unsigned*)(lv7330_reindex_shared_dyn_warp +
                                   (((int64_t)ax1_0_3) * (int64_t)8)))[3]),
                  "r"(((unsigned*)(p_output0_intermediate_1_reindex_shared_dyn_warp +
                                   (((int64_t)ax2_0_3) * (int64_t)8)))[0]),
                  "r"(((unsigned*)(p_output0_intermediate_1_reindex_shared_dyn_warp +
                                   (((int64_t)ax2_0_3) * (int64_t)8)))[1]),
                  "f"(((float*)(var_NT_matmul_intermediate_reindex_shared_dyn_warp +
                                ((((int64_t)ax1_0_3) * (int64_t)32) +
                                 (((int64_t)ax2_0_3) * (int64_t)8))))[0]),
                  "f"(((float*)(var_NT_matmul_intermediate_reindex_shared_dyn_warp +
                                ((((int64_t)ax1_0_3) * (int64_t)32) +
                                 (((int64_t)ax2_0_3) * (int64_t)8))))[1]),
                  "f"(((float*)(var_NT_matmul_intermediate_reindex_shared_dyn_warp +
                                ((((int64_t)ax1_0_3) * (int64_t)32) +
                                 (((int64_t)ax2_0_3) * (int64_t)8))))[2]),
                  "f"(((float*)(var_NT_matmul_intermediate_reindex_shared_dyn_warp +
                                ((((int64_t)ax1_0_3) * (int64_t)32) +
                                 (((int64_t)ax2_0_3) * (int64_t)8))))[3]));
          }

          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(((float*)(var_NT_matmul_intermediate_reindex_shared_dyn_warp +
                                 (((((int64_t)ax1_0_3) * (int64_t)32) +
                                   (((int64_t)ax2_0_3) * (int64_t)8)) +
                                  (int64_t)4)))[0]),
                  "=f"(((float*)(var_NT_matmul_intermediate_reindex_shared_dyn_warp +
                                 (((((int64_t)ax1_0_3) * (int64_t)32) +
                                   (((int64_t)ax2_0_3) * (int64_t)8)) +
                                  (int64_t)4)))[1]),
                  "=f"(((float*)(var_NT_matmul_intermediate_reindex_shared_dyn_warp +
                                 (((((int64_t)ax1_0_3) * (int64_t)32) +
                                   (((int64_t)ax2_0_3) * (int64_t)8)) +
                                  (int64_t)4)))[2]),
                  "=f"(((float*)(var_NT_matmul_intermediate_reindex_shared_dyn_warp +
                                 (((((int64_t)ax1_0_3) * (int64_t)32) +
                                   (((int64_t)ax2_0_3) * (int64_t)8)) +
                                  (int64_t)4)))[3])
                : "r"(((unsigned*)(lv7330_reindex_shared_dyn_warp +
                                   (((int64_t)ax1_0_3) * (int64_t)8)))[0]),
                  "r"(((unsigned*)(lv7330_reindex_shared_dyn_warp +
                                   (((int64_t)ax1_0_3) * (int64_t)8)))[1]),
                  "r"(((unsigned*)(lv7330_reindex_shared_dyn_warp +
                                   (((int64_t)ax1_0_3) * (int64_t)8)))[2]),
                  "r"(((unsigned*)(lv7330_reindex_shared_dyn_warp +
                                   (((int64_t)ax1_0_3) * (int64_t)8)))[3]),
                  "r"(((unsigned*)(p_output0_intermediate_1_reindex_shared_dyn_warp +
                                   ((((int64_t)ax2_0_3) * (int64_t)8) + (int64_t)4)))[0]),
                  "r"(((unsigned*)(p_output0_intermediate_1_reindex_shared_dyn_warp +
                                   ((((int64_t)ax2_0_3) * (int64_t)8) + (int64_t)4)))[1]),
                  "f"(((float*)(var_NT_matmul_intermediate_reindex_shared_dyn_warp +
                                (((((int64_t)ax1_0_3) * (int64_t)32) +
                                  (((int64_t)ax2_0_3) * (int64_t)8)) +
                                 (int64_t)4)))[0]),
                  "f"(((float*)(var_NT_matmul_intermediate_reindex_shared_dyn_warp +
                                (((((int64_t)ax1_0_3) * (int64_t)32) +
                                  (((int64_t)ax2_0_3) * (int64_t)8)) +
                                 (int64_t)4)))[1]),
                  "f"(((float*)(var_NT_matmul_intermediate_reindex_shared_dyn_warp +
                                (((((int64_t)ax1_0_3) * (int64_t)32) +
                                  (((int64_t)ax2_0_3) * (int64_t)8)) +
                                 (int64_t)4)))[2]),
                  "f"(((float*)(var_NT_matmul_intermediate_reindex_shared_dyn_warp +
                                (((((int64_t)ax1_0_3) * (int64_t)32) +
                                  (((int64_t)ax2_0_3) * (int64_t)8)) +
                                 (int64_t)4)))[3]));
          }
        }
      }
    }
  }
  __syncthreads();
  for (int64_t ax1_1 = 0; ax1_1 < (int64_t)4; ++ax1_1) {
    for (int64_t ax2_1 = 0; ax2_1 < (int64_t)4; ++ax2_1) {
      for (int64_t local_id = 0; local_id < (int64_t)8; ++local_id) {
        ((float*)buf_dyn_shmem)[(
            ((((((((int64_t)threadIdx.z) * (int64_t)8192) + (ax1_1 * (int64_t)2048)) +
                (((local_id & (int64_t)3) >> (int64_t)1) * (int64_t)1024)) +
               ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)128)) +
              (((((((int64_t)threadIdx.y) * (int64_t)8) + (ax2_1 * (int64_t)2)) +
                 (local_id >> (int64_t)2)) ^
                (((int64_t)threadIdx.x) >> (int64_t)2)) *
               (int64_t)8)) +
             ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)2)) +
            (local_id & (int64_t)1))] =
            var_NT_matmul_intermediate_reindex_shared_dyn_warp[(
                ((ax1_1 * (int64_t)32) + (ax2_1 * (int64_t)8)) + local_id)];
      }
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0_2 = 0; ax0_ax1_fused_0_2 < (int64_t)16; ++ax0_ax1_fused_0_2) {
    uint4 __1;
    ulonglong4 v_ = *(ulonglong4*)(((float*)buf_dyn_shmem) +
                                   (((((ax0_ax1_fused_0_2 * (int64_t)1024) +
                                       (((int64_t)threadIdx.z) * (int64_t)512)) +
                                      (((int64_t)threadIdx.y) * (int64_t)256)) +
                                     ((((int64_t)threadIdx.x) >> (int64_t)4) * (int64_t)128)) +
                                    (((((int64_t)threadIdx.x) & (int64_t)15) ^
                                      (((((int64_t)threadIdx.z) * (int64_t)4) +
                                        (((int64_t)threadIdx.y) * (int64_t)2)) +
                                       (((int64_t)threadIdx.x) >> (int64_t)4))) *
                                     (int64_t)8)));
    ((half2*)(&(__1.x)))->x = (half)(((float2*)(&(v_.x)))->x);
    ((half2*)(&(__1.x)))->y = (half)(((float2*)(&(v_.x)))->y);
    ((half2*)(&(__1.y)))->x = (half)(((float2*)(&(v_.y)))->x);
    ((half2*)(&(__1.y)))->y = (half)(((float2*)(&(v_.y)))->y);
    ((half2*)(&(__1.z)))->x = (half)(((float2*)(&(v_.z)))->x);
    ((half2*)(&(__1.z)))->y = (half)(((float2*)(&(v_.z)))->y);
    ((half2*)(&(__1.w)))->x = (half)(((float2*)(&(v_.w)))->x);
    ((half2*)(&(__1.w)))->y = (half)(((float2*)(&(v_.w)))->y);
    *(uint4*)(p_output0_intermediate +
              ((((((((((int64_t)blockIdx.x) / (int64_t)86) * (int64_t)1409024) +
                    (ax0_ax1_fused_0_2 * (int64_t)88064)) +
                   (((int64_t)threadIdx.z) * (int64_t)44032)) +
                  (((int64_t)threadIdx.y) * (int64_t)22016)) +
                 ((((int64_t)threadIdx.x) >> (int64_t)4) * (int64_t)11008)) +
                ((((int64_t)blockIdx.x) % (int64_t)86) * (int64_t)128)) +
               ((((int64_t)threadIdx.x) & (int64_t)15) * (int64_t)8))) = __1;
  }
}
#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

void fused_decode3_wrapper(torch::Tensor lv2608, torch::Tensor lv2609, torch::Tensor lv7330,
                           torch::Tensor p_output0_intermediate, int64_t b, int64_t blocksize) {
  uint* lv2608_data = (uint*)(lv2608.data_ptr());
  half* lv2609_data = (half*)(lv2609.data_ptr());
  half* lv7330_data = (half*)(lv7330.data_ptr());
  half* p_output0_data = (half*)(p_output0_intermediate.data_ptr());

  CUDA_CHECK(cudaFuncSetAttribute(fused_fused_decode3_fused_NT_matmul21_cast21_add5_silu2_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
  dim3 blocks(blocksize);  // 根据你的需求设置blocks
  dim3 threads(32, 2, 2);  // 根据你的需求设置threads
  fused_fused_decode3_fused_NT_matmul21_cast21_add5_silu2_kernel<<<blocks, threads, 65530>>>(
      lv2608_data, lv2609_data, lv7330_data, p_output0_data, b);
  cudaDeviceSynchronize();
}
