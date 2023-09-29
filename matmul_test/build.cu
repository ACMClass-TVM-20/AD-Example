#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#else

typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
#endif


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x) {          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 530)
CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)
#else
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hexp, exp)
#endif
#endif

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY
#include <mma.h>

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(128) default_function_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ compute) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> p_output0_intermediate_reindex_pad_shared_dyn_wmma_accumulator[16];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_reindex_shared_dyn_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> var_T_transpose_intermediate_reindex_pad_shared_dyn_wmma_matrix_b[4];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 4; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 4; ++ax2_0_3_init) {
      nvcuda::wmma::fill_fragment(p_output0_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_init * 4) + ax2_0_3_init)], 0.000000e+00f);
    }
  }
  #pragma unroll
  for (int ax3_0_0 = 0; ax3_0_0 < 128; ++ax3_0_0) {
    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0) {
      *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + (((((((((((int)blockIdx.x) % 288) / 18) * 524288) + (ax0_ax1_fused_0 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + (ax3_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 4; ++ax0_ax1_fused_0_1) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_1 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = (((((((int)blockIdx.x) / 288) * 9) + ((((int)blockIdx.x) % 18) >> 1)) < 43) ? *(uint4*)(B + (((((((((((int)blockIdx.x) / 288) * 9437184) + ((((int)blockIdx.x) % 18) * 524288)) + (ax0_ax1_fused_0_1 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + (ax3_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8))) : make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f))));
    }
    __syncthreads();
    for (int ax3_0_1 = 0; ax3_0_1 < 2; ++ax3_0_1) {
      for (int ax0 = 0; ax0 < 4; ++ax0) {
        nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0 * 768)) + (ax3_0_1 * 384)) + 6144)])), (int64_t)24);
      }
      for (int ax0_1 = 0; ax0_1 < 4; ++ax0_1) {
        nvcuda::wmma::load_matrix_sync(var_T_transpose_intermediate_reindex_pad_shared_dyn_wmma_matrix_b[ax0_1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_1 * 768)) + (ax3_0_1 * 384))])), (int64_t)24);
      }
      for (int ax1_0_3 = 0; ax1_0_3 < 4; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 4; ++ax2_0_3) {
          nvcuda::wmma::mma_sync(p_output0_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_3], var_T_transpose_intermediate_reindex_pad_shared_dyn_wmma_matrix_b[ax2_0_3], p_output0_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)]);
        }
      }
    }
  }
  __syncthreads();
  for (int ax2_1 = 0; ax2_1 < 4; ++ax2_1) {
    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
      nvcuda::wmma::store_matrix_sync((&(((float*)buf_dyn_shmem)[((((((int)threadIdx.z) * 8192) + (ax2_1 * 2048)) + (((int)threadIdx.y) * 1024)) + (ax3_1 * 256))])), p_output0_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax2_1 * 4) + ax3_1)], (int64_t)16, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_2 = 0; ax0_ax1_fused_0_2 < 16; ++ax0_ax1_fused_0_2) {
    if ((((((int)blockIdx.x) / 288) * 9) + ((((int)blockIdx.x) % 18) >> 1)) < 43) {
      uint4 __1;
      ulonglong4 v_ = *(ulonglong4*)(((float*)buf_dyn_shmem) + (((((((((ax0_ax1_fused_0_2 * 8) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) >> 4) * 2048) + (((((int)threadIdx.x) & 15) >> 1) * 256)) + ((((((ax0_ax1_fused_0_2 * 8) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) & 15) * 16)) + ((((int)threadIdx.x) & 1) * 8)));
      ((half2*)(&(__1.x)))->x = (half)(((float2*)(&(v_.x)))->x);
      ((half2*)(&(__1.x)))->y = (half)(((float2*)(&(v_.x)))->y);
      ((half2*)(&(__1.y)))->x = (half)(((float2*)(&(v_.y)))->x);
      ((half2*)(&(__1.y)))->y = (half)(((float2*)(&(v_.y)))->y);
      ((half2*)(&(__1.z)))->x = (half)(((float2*)(&(v_.z)))->x);
      ((half2*)(&(__1.z)))->y = (half)(((float2*)(&(v_.z)))->y);
      ((half2*)(&(__1.w)))->x = (half)(((float2*)(&(v_.w)))->x);
      ((half2*)(&(__1.w)))->y = (half)(((float2*)(&(v_.w)))->y);
      *(uint4*)(compute + ((((((((((((int)blockIdx.x) % 288) / 18) * 1409024) + (ax0_ax1_fused_0_2 * 88064)) + (((int)threadIdx.z) * 44032)) + (((int)threadIdx.y) * 22016)) + ((((int)threadIdx.x) >> 4) * 11008)) + ((((int)blockIdx.x) / 288) * 2304)) + ((((int)blockIdx.x) % 18) * 128)) + ((((int)threadIdx.x) & 15) * 8))) = __1;
    }
  }
}

