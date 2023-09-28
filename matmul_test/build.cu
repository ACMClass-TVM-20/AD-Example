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
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> matmul_reindex_shared_dyn_wmma_accumulator[16];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_reindex_shared_dyn_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> T_transpose_reindex_shared_dyn_wmma_matrix_b[4];
  for (int ax1_0_2_init = 0; ax1_0_2_init < 4; ++ax1_0_2_init) {
    for (int ax2_0_2_init = 0; ax2_0_2_init < 4; ++ax2_0_2_init) {
      nvcuda::wmma::fill_fragment(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_init * 4) + ax2_0_2_init)], 0.000000e+00f);
    }
  }
  for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + (((((ax0_ax1_fused_0 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)));
  }
  for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 4; ++ax0_ax1_fused_0_1) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_1 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + ((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_1 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)));
  }
  __syncthreads();
  for (int ax3_0_1 = 0; ax3_0_1 < 2; ++ax3_0_1) {
    for (int ax0 = 0; ax0 < 4; ++ax0) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0 * 768)) + (ax3_0_1 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_1 = 0; ax0_1 < 4; ++ax0_1) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_1 * 768)) + (ax3_0_1 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2 = 0; ax1_0_2 < 4; ++ax1_0_2) {
      for (int ax2_0_2 = 0; ax2_0_2 < 4; ++ax2_0_2) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2 * 4) + ax2_0_2)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2 * 4) + ax2_0_2)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_2 = 0; ax0_ax1_fused_0_2 < 4; ++ax0_ax1_fused_0_2) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_2 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_2 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 32));
  }
  for (int ax0_ax1_fused_0_3 = 0; ax0_ax1_fused_0_3 < 4; ++ax0_ax1_fused_0_3) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_3 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_3 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 32));
  }
  __syncthreads();
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 2; ++ax3_0_1_1) {
    for (int ax0_2 = 0; ax0_2 < 4; ++ax0_2) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_2], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_2 * 768)) + (ax3_0_1_1 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_3 = 0; ax0_3 < 4; ++ax0_3) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_3 * 768)) + (ax3_0_1_1 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_1 = 0; ax1_0_2_1 < 4; ++ax1_0_2_1) {
      for (int ax2_0_2_1 = 0; ax2_0_2_1 < 4; ++ax2_0_2_1) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_1 * 4) + ax2_0_2_1)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_1], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_1], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_1 * 4) + ax2_0_2_1)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_4 = 0; ax0_ax1_fused_0_4 < 4; ++ax0_ax1_fused_0_4) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_4 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_4 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 64));
  }
  for (int ax0_ax1_fused_0_5 = 0; ax0_ax1_fused_0_5 < 4; ++ax0_ax1_fused_0_5) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_5 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_5 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 64));
  }
  __syncthreads();
  for (int ax3_0_1_2 = 0; ax3_0_1_2 < 2; ++ax3_0_1_2) {
    for (int ax0_4 = 0; ax0_4 < 4; ++ax0_4) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_4], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_4 * 768)) + (ax3_0_1_2 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_5 = 0; ax0_5 < 4; ++ax0_5) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_5], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_5 * 768)) + (ax3_0_1_2 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_2 = 0; ax1_0_2_2 < 4; ++ax1_0_2_2) {
      for (int ax2_0_2_2 = 0; ax2_0_2_2 < 4; ++ax2_0_2_2) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_2 * 4) + ax2_0_2_2)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_2], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_2], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_2 * 4) + ax2_0_2_2)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_6 = 0; ax0_ax1_fused_0_6 < 4; ++ax0_ax1_fused_0_6) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_6 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_6 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 96));
  }
  for (int ax0_ax1_fused_0_7 = 0; ax0_ax1_fused_0_7 < 4; ++ax0_ax1_fused_0_7) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_7 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_7 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 96));
  }
  __syncthreads();
  for (int ax3_0_1_3 = 0; ax3_0_1_3 < 2; ++ax3_0_1_3) {
    for (int ax0_6 = 0; ax0_6 < 4; ++ax0_6) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_6], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_6 * 768)) + (ax3_0_1_3 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_7 = 0; ax0_7 < 4; ++ax0_7) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_7], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_7 * 768)) + (ax3_0_1_3 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_3 = 0; ax1_0_2_3 < 4; ++ax1_0_2_3) {
      for (int ax2_0_2_3 = 0; ax2_0_2_3 < 4; ++ax2_0_2_3) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_3 * 4) + ax2_0_2_3)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_3], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_3], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_3 * 4) + ax2_0_2_3)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_8 = 0; ax0_ax1_fused_0_8 < 4; ++ax0_ax1_fused_0_8) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_8 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_8 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 128));
  }
  for (int ax0_ax1_fused_0_9 = 0; ax0_ax1_fused_0_9 < 4; ++ax0_ax1_fused_0_9) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_9 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_9 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 128));
  }
  __syncthreads();
  for (int ax3_0_1_4 = 0; ax3_0_1_4 < 2; ++ax3_0_1_4) {
    for (int ax0_8 = 0; ax0_8 < 4; ++ax0_8) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_8], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_8 * 768)) + (ax3_0_1_4 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_9 = 0; ax0_9 < 4; ++ax0_9) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_9], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_9 * 768)) + (ax3_0_1_4 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_4 = 0; ax1_0_2_4 < 4; ++ax1_0_2_4) {
      for (int ax2_0_2_4 = 0; ax2_0_2_4 < 4; ++ax2_0_2_4) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_4 * 4) + ax2_0_2_4)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_4], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_4], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_4 * 4) + ax2_0_2_4)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_10 = 0; ax0_ax1_fused_0_10 < 4; ++ax0_ax1_fused_0_10) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_10 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_10 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 160));
  }
  for (int ax0_ax1_fused_0_11 = 0; ax0_ax1_fused_0_11 < 4; ++ax0_ax1_fused_0_11) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_11 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_11 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 160));
  }
  __syncthreads();
  for (int ax3_0_1_5 = 0; ax3_0_1_5 < 2; ++ax3_0_1_5) {
    for (int ax0_10 = 0; ax0_10 < 4; ++ax0_10) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_10], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_10 * 768)) + (ax3_0_1_5 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_11 = 0; ax0_11 < 4; ++ax0_11) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_11], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_11 * 768)) + (ax3_0_1_5 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_5 = 0; ax1_0_2_5 < 4; ++ax1_0_2_5) {
      for (int ax2_0_2_5 = 0; ax2_0_2_5 < 4; ++ax2_0_2_5) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_5 * 4) + ax2_0_2_5)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_5], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_5], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_5 * 4) + ax2_0_2_5)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_12 = 0; ax0_ax1_fused_0_12 < 4; ++ax0_ax1_fused_0_12) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_12 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_12 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 192));
  }
  for (int ax0_ax1_fused_0_13 = 0; ax0_ax1_fused_0_13 < 4; ++ax0_ax1_fused_0_13) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_13 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_13 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 192));
  }
  __syncthreads();
  for (int ax3_0_1_6 = 0; ax3_0_1_6 < 2; ++ax3_0_1_6) {
    for (int ax0_12 = 0; ax0_12 < 4; ++ax0_12) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_12], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_12 * 768)) + (ax3_0_1_6 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_13 = 0; ax0_13 < 4; ++ax0_13) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_13], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_13 * 768)) + (ax3_0_1_6 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_6 = 0; ax1_0_2_6 < 4; ++ax1_0_2_6) {
      for (int ax2_0_2_6 = 0; ax2_0_2_6 < 4; ++ax2_0_2_6) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_6 * 4) + ax2_0_2_6)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_6], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_6], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_6 * 4) + ax2_0_2_6)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_14 = 0; ax0_ax1_fused_0_14 < 4; ++ax0_ax1_fused_0_14) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_14 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_14 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 224));
  }
  for (int ax0_ax1_fused_0_15 = 0; ax0_ax1_fused_0_15 < 4; ++ax0_ax1_fused_0_15) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_15 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_15 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 224));
  }
  __syncthreads();
  for (int ax3_0_1_7 = 0; ax3_0_1_7 < 2; ++ax3_0_1_7) {
    for (int ax0_14 = 0; ax0_14 < 4; ++ax0_14) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_14], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_14 * 768)) + (ax3_0_1_7 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_15 = 0; ax0_15 < 4; ++ax0_15) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_15], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_15 * 768)) + (ax3_0_1_7 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_7 = 0; ax1_0_2_7 < 4; ++ax1_0_2_7) {
      for (int ax2_0_2_7 = 0; ax2_0_2_7 < 4; ++ax2_0_2_7) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_7 * 4) + ax2_0_2_7)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_7], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_7], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_7 * 4) + ax2_0_2_7)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_16 = 0; ax0_ax1_fused_0_16 < 4; ++ax0_ax1_fused_0_16) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_16 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_16 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 256));
  }
  for (int ax0_ax1_fused_0_17 = 0; ax0_ax1_fused_0_17 < 4; ++ax0_ax1_fused_0_17) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_17 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_17 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 256));
  }
  __syncthreads();
  for (int ax3_0_1_8 = 0; ax3_0_1_8 < 2; ++ax3_0_1_8) {
    for (int ax0_16 = 0; ax0_16 < 4; ++ax0_16) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_16], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_16 * 768)) + (ax3_0_1_8 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_17 = 0; ax0_17 < 4; ++ax0_17) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_17], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_17 * 768)) + (ax3_0_1_8 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_8 = 0; ax1_0_2_8 < 4; ++ax1_0_2_8) {
      for (int ax2_0_2_8 = 0; ax2_0_2_8 < 4; ++ax2_0_2_8) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_8 * 4) + ax2_0_2_8)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_8], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_8], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_8 * 4) + ax2_0_2_8)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_18 = 0; ax0_ax1_fused_0_18 < 4; ++ax0_ax1_fused_0_18) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_18 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_18 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 288));
  }
  for (int ax0_ax1_fused_0_19 = 0; ax0_ax1_fused_0_19 < 4; ++ax0_ax1_fused_0_19) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_19 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_19 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 288));
  }
  __syncthreads();
  for (int ax3_0_1_9 = 0; ax3_0_1_9 < 2; ++ax3_0_1_9) {
    for (int ax0_18 = 0; ax0_18 < 4; ++ax0_18) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_18], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_18 * 768)) + (ax3_0_1_9 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_19 = 0; ax0_19 < 4; ++ax0_19) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_19], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_19 * 768)) + (ax3_0_1_9 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_9 = 0; ax1_0_2_9 < 4; ++ax1_0_2_9) {
      for (int ax2_0_2_9 = 0; ax2_0_2_9 < 4; ++ax2_0_2_9) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_9 * 4) + ax2_0_2_9)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_9], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_9], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_9 * 4) + ax2_0_2_9)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_20 = 0; ax0_ax1_fused_0_20 < 4; ++ax0_ax1_fused_0_20) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_20 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_20 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 320));
  }
  for (int ax0_ax1_fused_0_21 = 0; ax0_ax1_fused_0_21 < 4; ++ax0_ax1_fused_0_21) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_21 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_21 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 320));
  }
  __syncthreads();
  for (int ax3_0_1_10 = 0; ax3_0_1_10 < 2; ++ax3_0_1_10) {
    for (int ax0_20 = 0; ax0_20 < 4; ++ax0_20) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_20], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_20 * 768)) + (ax3_0_1_10 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_21 = 0; ax0_21 < 4; ++ax0_21) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_21], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_21 * 768)) + (ax3_0_1_10 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_10 = 0; ax1_0_2_10 < 4; ++ax1_0_2_10) {
      for (int ax2_0_2_10 = 0; ax2_0_2_10 < 4; ++ax2_0_2_10) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_10 * 4) + ax2_0_2_10)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_10], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_10], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_10 * 4) + ax2_0_2_10)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_22 = 0; ax0_ax1_fused_0_22 < 4; ++ax0_ax1_fused_0_22) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_22 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_22 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 352));
  }
  for (int ax0_ax1_fused_0_23 = 0; ax0_ax1_fused_0_23 < 4; ++ax0_ax1_fused_0_23) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_23 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_23 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 352));
  }
  __syncthreads();
  for (int ax3_0_1_11 = 0; ax3_0_1_11 < 2; ++ax3_0_1_11) {
    for (int ax0_22 = 0; ax0_22 < 4; ++ax0_22) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_22], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_22 * 768)) + (ax3_0_1_11 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_23 = 0; ax0_23 < 4; ++ax0_23) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_23], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_23 * 768)) + (ax3_0_1_11 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_11 = 0; ax1_0_2_11 < 4; ++ax1_0_2_11) {
      for (int ax2_0_2_11 = 0; ax2_0_2_11 < 4; ++ax2_0_2_11) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_11 * 4) + ax2_0_2_11)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_11], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_11], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_11 * 4) + ax2_0_2_11)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_24 = 0; ax0_ax1_fused_0_24 < 4; ++ax0_ax1_fused_0_24) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_24 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_24 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 384));
  }
  for (int ax0_ax1_fused_0_25 = 0; ax0_ax1_fused_0_25 < 4; ++ax0_ax1_fused_0_25) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_25 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_25 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 384));
  }
  __syncthreads();
  for (int ax3_0_1_12 = 0; ax3_0_1_12 < 2; ++ax3_0_1_12) {
    for (int ax0_24 = 0; ax0_24 < 4; ++ax0_24) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_24], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_24 * 768)) + (ax3_0_1_12 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_25 = 0; ax0_25 < 4; ++ax0_25) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_25], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_25 * 768)) + (ax3_0_1_12 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_12 = 0; ax1_0_2_12 < 4; ++ax1_0_2_12) {
      for (int ax2_0_2_12 = 0; ax2_0_2_12 < 4; ++ax2_0_2_12) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_12 * 4) + ax2_0_2_12)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_12], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_12], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_12 * 4) + ax2_0_2_12)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_26 = 0; ax0_ax1_fused_0_26 < 4; ++ax0_ax1_fused_0_26) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_26 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_26 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 416));
  }
  for (int ax0_ax1_fused_0_27 = 0; ax0_ax1_fused_0_27 < 4; ++ax0_ax1_fused_0_27) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_27 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_27 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 416));
  }
  __syncthreads();
  for (int ax3_0_1_13 = 0; ax3_0_1_13 < 2; ++ax3_0_1_13) {
    for (int ax0_26 = 0; ax0_26 < 4; ++ax0_26) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_26], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_26 * 768)) + (ax3_0_1_13 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_27 = 0; ax0_27 < 4; ++ax0_27) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_27], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_27 * 768)) + (ax3_0_1_13 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_13 = 0; ax1_0_2_13 < 4; ++ax1_0_2_13) {
      for (int ax2_0_2_13 = 0; ax2_0_2_13 < 4; ++ax2_0_2_13) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_13 * 4) + ax2_0_2_13)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_13], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_13], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_13 * 4) + ax2_0_2_13)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_28 = 0; ax0_ax1_fused_0_28 < 4; ++ax0_ax1_fused_0_28) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_28 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_28 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 448));
  }
  for (int ax0_ax1_fused_0_29 = 0; ax0_ax1_fused_0_29 < 4; ++ax0_ax1_fused_0_29) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_29 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_29 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 448));
  }
  __syncthreads();
  for (int ax3_0_1_14 = 0; ax3_0_1_14 < 2; ++ax3_0_1_14) {
    for (int ax0_28 = 0; ax0_28 < 4; ++ax0_28) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_28], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_28 * 768)) + (ax3_0_1_14 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_29 = 0; ax0_29 < 4; ++ax0_29) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_29], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_29 * 768)) + (ax3_0_1_14 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_14 = 0; ax1_0_2_14 < 4; ++ax1_0_2_14) {
      for (int ax2_0_2_14 = 0; ax2_0_2_14 < 4; ++ax2_0_2_14) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_14 * 4) + ax2_0_2_14)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_14], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_14], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_14 * 4) + ax2_0_2_14)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_30 = 0; ax0_ax1_fused_0_30 < 4; ++ax0_ax1_fused_0_30) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_30 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_30 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 480));
  }
  for (int ax0_ax1_fused_0_31 = 0; ax0_ax1_fused_0_31 < 4; ++ax0_ax1_fused_0_31) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_31 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_31 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 480));
  }
  __syncthreads();
  for (int ax3_0_1_15 = 0; ax3_0_1_15 < 2; ++ax3_0_1_15) {
    for (int ax0_30 = 0; ax0_30 < 4; ++ax0_30) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_30], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_30 * 768)) + (ax3_0_1_15 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_31 = 0; ax0_31 < 4; ++ax0_31) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_31], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_31 * 768)) + (ax3_0_1_15 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_15 = 0; ax1_0_2_15 < 4; ++ax1_0_2_15) {
      for (int ax2_0_2_15 = 0; ax2_0_2_15 < 4; ++ax2_0_2_15) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_15 * 4) + ax2_0_2_15)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_15], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_15], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_15 * 4) + ax2_0_2_15)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_32 = 0; ax0_ax1_fused_0_32 < 4; ++ax0_ax1_fused_0_32) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_32 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_32 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 512));
  }
  for (int ax0_ax1_fused_0_33 = 0; ax0_ax1_fused_0_33 < 4; ++ax0_ax1_fused_0_33) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_33 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_33 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 512));
  }
  __syncthreads();
  for (int ax3_0_1_16 = 0; ax3_0_1_16 < 2; ++ax3_0_1_16) {
    for (int ax0_32 = 0; ax0_32 < 4; ++ax0_32) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_32], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_32 * 768)) + (ax3_0_1_16 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_33 = 0; ax0_33 < 4; ++ax0_33) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_33], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_33 * 768)) + (ax3_0_1_16 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_16 = 0; ax1_0_2_16 < 4; ++ax1_0_2_16) {
      for (int ax2_0_2_16 = 0; ax2_0_2_16 < 4; ++ax2_0_2_16) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_16 * 4) + ax2_0_2_16)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_16], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_16], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_16 * 4) + ax2_0_2_16)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_34 = 0; ax0_ax1_fused_0_34 < 4; ++ax0_ax1_fused_0_34) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_34 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_34 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 544));
  }
  for (int ax0_ax1_fused_0_35 = 0; ax0_ax1_fused_0_35 < 4; ++ax0_ax1_fused_0_35) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_35 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_35 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 544));
  }
  __syncthreads();
  for (int ax3_0_1_17 = 0; ax3_0_1_17 < 2; ++ax3_0_1_17) {
    for (int ax0_34 = 0; ax0_34 < 4; ++ax0_34) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_34], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_34 * 768)) + (ax3_0_1_17 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_35 = 0; ax0_35 < 4; ++ax0_35) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_35], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_35 * 768)) + (ax3_0_1_17 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_17 = 0; ax1_0_2_17 < 4; ++ax1_0_2_17) {
      for (int ax2_0_2_17 = 0; ax2_0_2_17 < 4; ++ax2_0_2_17) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_17 * 4) + ax2_0_2_17)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_17], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_17], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_17 * 4) + ax2_0_2_17)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_36 = 0; ax0_ax1_fused_0_36 < 4; ++ax0_ax1_fused_0_36) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_36 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_36 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 576));
  }
  for (int ax0_ax1_fused_0_37 = 0; ax0_ax1_fused_0_37 < 4; ++ax0_ax1_fused_0_37) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_37 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_37 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 576));
  }
  __syncthreads();
  for (int ax3_0_1_18 = 0; ax3_0_1_18 < 2; ++ax3_0_1_18) {
    for (int ax0_36 = 0; ax0_36 < 4; ++ax0_36) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_36], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_36 * 768)) + (ax3_0_1_18 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_37 = 0; ax0_37 < 4; ++ax0_37) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_37], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_37 * 768)) + (ax3_0_1_18 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_18 = 0; ax1_0_2_18 < 4; ++ax1_0_2_18) {
      for (int ax2_0_2_18 = 0; ax2_0_2_18 < 4; ++ax2_0_2_18) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_18 * 4) + ax2_0_2_18)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_18], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_18], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_18 * 4) + ax2_0_2_18)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_38 = 0; ax0_ax1_fused_0_38 < 4; ++ax0_ax1_fused_0_38) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_38 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_38 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 608));
  }
  for (int ax0_ax1_fused_0_39 = 0; ax0_ax1_fused_0_39 < 4; ++ax0_ax1_fused_0_39) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_39 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_39 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 608));
  }
  __syncthreads();
  for (int ax3_0_1_19 = 0; ax3_0_1_19 < 2; ++ax3_0_1_19) {
    for (int ax0_38 = 0; ax0_38 < 4; ++ax0_38) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_38], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_38 * 768)) + (ax3_0_1_19 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_39 = 0; ax0_39 < 4; ++ax0_39) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_39], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_39 * 768)) + (ax3_0_1_19 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_19 = 0; ax1_0_2_19 < 4; ++ax1_0_2_19) {
      for (int ax2_0_2_19 = 0; ax2_0_2_19 < 4; ++ax2_0_2_19) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_19 * 4) + ax2_0_2_19)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_19], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_19], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_19 * 4) + ax2_0_2_19)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_40 = 0; ax0_ax1_fused_0_40 < 4; ++ax0_ax1_fused_0_40) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_40 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_40 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 640));
  }
  for (int ax0_ax1_fused_0_41 = 0; ax0_ax1_fused_0_41 < 4; ++ax0_ax1_fused_0_41) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_41 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_41 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 640));
  }
  __syncthreads();
  for (int ax3_0_1_20 = 0; ax3_0_1_20 < 2; ++ax3_0_1_20) {
    for (int ax0_40 = 0; ax0_40 < 4; ++ax0_40) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_40], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_40 * 768)) + (ax3_0_1_20 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_41 = 0; ax0_41 < 4; ++ax0_41) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_41], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_41 * 768)) + (ax3_0_1_20 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_20 = 0; ax1_0_2_20 < 4; ++ax1_0_2_20) {
      for (int ax2_0_2_20 = 0; ax2_0_2_20 < 4; ++ax2_0_2_20) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_20 * 4) + ax2_0_2_20)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_20], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_20], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_20 * 4) + ax2_0_2_20)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_42 = 0; ax0_ax1_fused_0_42 < 4; ++ax0_ax1_fused_0_42) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_42 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_42 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 672));
  }
  for (int ax0_ax1_fused_0_43 = 0; ax0_ax1_fused_0_43 < 4; ++ax0_ax1_fused_0_43) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_43 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_43 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 672));
  }
  __syncthreads();
  for (int ax3_0_1_21 = 0; ax3_0_1_21 < 2; ++ax3_0_1_21) {
    for (int ax0_42 = 0; ax0_42 < 4; ++ax0_42) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_42], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_42 * 768)) + (ax3_0_1_21 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_43 = 0; ax0_43 < 4; ++ax0_43) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_43], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_43 * 768)) + (ax3_0_1_21 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_21 = 0; ax1_0_2_21 < 4; ++ax1_0_2_21) {
      for (int ax2_0_2_21 = 0; ax2_0_2_21 < 4; ++ax2_0_2_21) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_21 * 4) + ax2_0_2_21)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_21], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_21], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_21 * 4) + ax2_0_2_21)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_44 = 0; ax0_ax1_fused_0_44 < 4; ++ax0_ax1_fused_0_44) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_44 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_44 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 704));
  }
  for (int ax0_ax1_fused_0_45 = 0; ax0_ax1_fused_0_45 < 4; ++ax0_ax1_fused_0_45) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_45 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_45 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 704));
  }
  __syncthreads();
  for (int ax3_0_1_22 = 0; ax3_0_1_22 < 2; ++ax3_0_1_22) {
    for (int ax0_44 = 0; ax0_44 < 4; ++ax0_44) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_44], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_44 * 768)) + (ax3_0_1_22 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_45 = 0; ax0_45 < 4; ++ax0_45) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_45], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_45 * 768)) + (ax3_0_1_22 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_22 = 0; ax1_0_2_22 < 4; ++ax1_0_2_22) {
      for (int ax2_0_2_22 = 0; ax2_0_2_22 < 4; ++ax2_0_2_22) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_22 * 4) + ax2_0_2_22)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_22], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_22], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_22 * 4) + ax2_0_2_22)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_46 = 0; ax0_ax1_fused_0_46 < 4; ++ax0_ax1_fused_0_46) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_46 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_46 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 736));
  }
  for (int ax0_ax1_fused_0_47 = 0; ax0_ax1_fused_0_47 < 4; ++ax0_ax1_fused_0_47) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_47 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_47 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 736));
  }
  __syncthreads();
  for (int ax3_0_1_23 = 0; ax3_0_1_23 < 2; ++ax3_0_1_23) {
    for (int ax0_46 = 0; ax0_46 < 4; ++ax0_46) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_46], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_46 * 768)) + (ax3_0_1_23 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_47 = 0; ax0_47 < 4; ++ax0_47) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_47], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_47 * 768)) + (ax3_0_1_23 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_23 = 0; ax1_0_2_23 < 4; ++ax1_0_2_23) {
      for (int ax2_0_2_23 = 0; ax2_0_2_23 < 4; ++ax2_0_2_23) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_23 * 4) + ax2_0_2_23)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_23], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_23], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_23 * 4) + ax2_0_2_23)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_48 = 0; ax0_ax1_fused_0_48 < 4; ++ax0_ax1_fused_0_48) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_48 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_48 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 768));
  }
  for (int ax0_ax1_fused_0_49 = 0; ax0_ax1_fused_0_49 < 4; ++ax0_ax1_fused_0_49) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_49 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_49 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 768));
  }
  __syncthreads();
  for (int ax3_0_1_24 = 0; ax3_0_1_24 < 2; ++ax3_0_1_24) {
    for (int ax0_48 = 0; ax0_48 < 4; ++ax0_48) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_48], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_48 * 768)) + (ax3_0_1_24 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_49 = 0; ax0_49 < 4; ++ax0_49) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_49], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_49 * 768)) + (ax3_0_1_24 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_24 = 0; ax1_0_2_24 < 4; ++ax1_0_2_24) {
      for (int ax2_0_2_24 = 0; ax2_0_2_24 < 4; ++ax2_0_2_24) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_24 * 4) + ax2_0_2_24)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_24], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_24], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_24 * 4) + ax2_0_2_24)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_50 = 0; ax0_ax1_fused_0_50 < 4; ++ax0_ax1_fused_0_50) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_50 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_50 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 800));
  }
  for (int ax0_ax1_fused_0_51 = 0; ax0_ax1_fused_0_51 < 4; ++ax0_ax1_fused_0_51) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_51 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_51 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 800));
  }
  __syncthreads();
  for (int ax3_0_1_25 = 0; ax3_0_1_25 < 2; ++ax3_0_1_25) {
    for (int ax0_50 = 0; ax0_50 < 4; ++ax0_50) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_50], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_50 * 768)) + (ax3_0_1_25 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_51 = 0; ax0_51 < 4; ++ax0_51) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_51], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_51 * 768)) + (ax3_0_1_25 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_25 = 0; ax1_0_2_25 < 4; ++ax1_0_2_25) {
      for (int ax2_0_2_25 = 0; ax2_0_2_25 < 4; ++ax2_0_2_25) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_25 * 4) + ax2_0_2_25)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_25], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_25], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_25 * 4) + ax2_0_2_25)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_52 = 0; ax0_ax1_fused_0_52 < 4; ++ax0_ax1_fused_0_52) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_52 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_52 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 832));
  }
  for (int ax0_ax1_fused_0_53 = 0; ax0_ax1_fused_0_53 < 4; ++ax0_ax1_fused_0_53) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_53 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_53 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 832));
  }
  __syncthreads();
  for (int ax3_0_1_26 = 0; ax3_0_1_26 < 2; ++ax3_0_1_26) {
    for (int ax0_52 = 0; ax0_52 < 4; ++ax0_52) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_52], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_52 * 768)) + (ax3_0_1_26 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_53 = 0; ax0_53 < 4; ++ax0_53) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_53], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_53 * 768)) + (ax3_0_1_26 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_26 = 0; ax1_0_2_26 < 4; ++ax1_0_2_26) {
      for (int ax2_0_2_26 = 0; ax2_0_2_26 < 4; ++ax2_0_2_26) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_26 * 4) + ax2_0_2_26)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_26], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_26], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_26 * 4) + ax2_0_2_26)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_54 = 0; ax0_ax1_fused_0_54 < 4; ++ax0_ax1_fused_0_54) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_54 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_54 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 864));
  }
  for (int ax0_ax1_fused_0_55 = 0; ax0_ax1_fused_0_55 < 4; ++ax0_ax1_fused_0_55) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_55 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_55 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 864));
  }
  __syncthreads();
  for (int ax3_0_1_27 = 0; ax3_0_1_27 < 2; ++ax3_0_1_27) {
    for (int ax0_54 = 0; ax0_54 < 4; ++ax0_54) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_54], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_54 * 768)) + (ax3_0_1_27 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_55 = 0; ax0_55 < 4; ++ax0_55) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_55], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_55 * 768)) + (ax3_0_1_27 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_27 = 0; ax1_0_2_27 < 4; ++ax1_0_2_27) {
      for (int ax2_0_2_27 = 0; ax2_0_2_27 < 4; ++ax2_0_2_27) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_27 * 4) + ax2_0_2_27)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_27], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_27], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_27 * 4) + ax2_0_2_27)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_56 = 0; ax0_ax1_fused_0_56 < 4; ++ax0_ax1_fused_0_56) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_56 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_56 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 896));
  }
  for (int ax0_ax1_fused_0_57 = 0; ax0_ax1_fused_0_57 < 4; ++ax0_ax1_fused_0_57) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_57 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_57 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 896));
  }
  __syncthreads();
  for (int ax3_0_1_28 = 0; ax3_0_1_28 < 2; ++ax3_0_1_28) {
    for (int ax0_56 = 0; ax0_56 < 4; ++ax0_56) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_56], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_56 * 768)) + (ax3_0_1_28 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_57 = 0; ax0_57 < 4; ++ax0_57) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_57], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_57 * 768)) + (ax3_0_1_28 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_28 = 0; ax1_0_2_28 < 4; ++ax1_0_2_28) {
      for (int ax2_0_2_28 = 0; ax2_0_2_28 < 4; ++ax2_0_2_28) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_28 * 4) + ax2_0_2_28)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_28], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_28], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_28 * 4) + ax2_0_2_28)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_58 = 0; ax0_ax1_fused_0_58 < 4; ++ax0_ax1_fused_0_58) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_58 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_58 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 928));
  }
  for (int ax0_ax1_fused_0_59 = 0; ax0_ax1_fused_0_59 < 4; ++ax0_ax1_fused_0_59) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_59 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_59 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 928));
  }
  __syncthreads();
  for (int ax3_0_1_29 = 0; ax3_0_1_29 < 2; ++ax3_0_1_29) {
    for (int ax0_58 = 0; ax0_58 < 4; ++ax0_58) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_58], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_58 * 768)) + (ax3_0_1_29 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_59 = 0; ax0_59 < 4; ++ax0_59) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_59], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_59 * 768)) + (ax3_0_1_29 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_29 = 0; ax1_0_2_29 < 4; ++ax1_0_2_29) {
      for (int ax2_0_2_29 = 0; ax2_0_2_29 < 4; ++ax2_0_2_29) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_29 * 4) + ax2_0_2_29)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_29], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_29], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_29 * 4) + ax2_0_2_29)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_60 = 0; ax0_ax1_fused_0_60 < 4; ++ax0_ax1_fused_0_60) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_60 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_60 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 960));
  }
  for (int ax0_ax1_fused_0_61 = 0; ax0_ax1_fused_0_61 < 4; ++ax0_ax1_fused_0_61) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_61 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_61 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 960));
  }
  __syncthreads();
  for (int ax3_0_1_30 = 0; ax3_0_1_30 < 2; ++ax3_0_1_30) {
    for (int ax0_60 = 0; ax0_60 < 4; ++ax0_60) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_60], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_60 * 768)) + (ax3_0_1_30 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_61 = 0; ax0_61 < 4; ++ax0_61) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_61], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_61 * 768)) + (ax3_0_1_30 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_30 = 0; ax1_0_2_30 < 4; ++ax1_0_2_30) {
      for (int ax2_0_2_30 = 0; ax2_0_2_30 < 4; ++ax2_0_2_30) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_30 * 4) + ax2_0_2_30)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_30], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_30], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_30 * 4) + ax2_0_2_30)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_62 = 0; ax0_ax1_fused_0_62 < 4; ++ax0_ax1_fused_0_62) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_62 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_62 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 992));
  }
  for (int ax0_ax1_fused_0_63 = 0; ax0_ax1_fused_0_63 < 4; ++ax0_ax1_fused_0_63) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_63 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_63 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 992));
  }
  __syncthreads();
  for (int ax3_0_1_31 = 0; ax3_0_1_31 < 2; ++ax3_0_1_31) {
    for (int ax0_62 = 0; ax0_62 < 4; ++ax0_62) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_62], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_62 * 768)) + (ax3_0_1_31 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_63 = 0; ax0_63 < 4; ++ax0_63) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_63], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_63 * 768)) + (ax3_0_1_31 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_31 = 0; ax1_0_2_31 < 4; ++ax1_0_2_31) {
      for (int ax2_0_2_31 = 0; ax2_0_2_31 < 4; ++ax2_0_2_31) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_31 * 4) + ax2_0_2_31)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_31], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_31], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_31 * 4) + ax2_0_2_31)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_64 = 0; ax0_ax1_fused_0_64 < 4; ++ax0_ax1_fused_0_64) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_64 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_64 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1024));
  }
  for (int ax0_ax1_fused_0_65 = 0; ax0_ax1_fused_0_65 < 4; ++ax0_ax1_fused_0_65) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_65 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_65 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1024));
  }
  __syncthreads();
  for (int ax3_0_1_32 = 0; ax3_0_1_32 < 2; ++ax3_0_1_32) {
    for (int ax0_64 = 0; ax0_64 < 4; ++ax0_64) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_64], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_64 * 768)) + (ax3_0_1_32 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_65 = 0; ax0_65 < 4; ++ax0_65) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_65], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_65 * 768)) + (ax3_0_1_32 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_32 = 0; ax1_0_2_32 < 4; ++ax1_0_2_32) {
      for (int ax2_0_2_32 = 0; ax2_0_2_32 < 4; ++ax2_0_2_32) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_32 * 4) + ax2_0_2_32)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_32], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_32], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_32 * 4) + ax2_0_2_32)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_66 = 0; ax0_ax1_fused_0_66 < 4; ++ax0_ax1_fused_0_66) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_66 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_66 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1056));
  }
  for (int ax0_ax1_fused_0_67 = 0; ax0_ax1_fused_0_67 < 4; ++ax0_ax1_fused_0_67) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_67 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_67 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1056));
  }
  __syncthreads();
  for (int ax3_0_1_33 = 0; ax3_0_1_33 < 2; ++ax3_0_1_33) {
    for (int ax0_66 = 0; ax0_66 < 4; ++ax0_66) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_66], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_66 * 768)) + (ax3_0_1_33 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_67 = 0; ax0_67 < 4; ++ax0_67) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_67], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_67 * 768)) + (ax3_0_1_33 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_33 = 0; ax1_0_2_33 < 4; ++ax1_0_2_33) {
      for (int ax2_0_2_33 = 0; ax2_0_2_33 < 4; ++ax2_0_2_33) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_33 * 4) + ax2_0_2_33)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_33], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_33], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_33 * 4) + ax2_0_2_33)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_68 = 0; ax0_ax1_fused_0_68 < 4; ++ax0_ax1_fused_0_68) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_68 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_68 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1088));
  }
  for (int ax0_ax1_fused_0_69 = 0; ax0_ax1_fused_0_69 < 4; ++ax0_ax1_fused_0_69) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_69 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_69 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1088));
  }
  __syncthreads();
  for (int ax3_0_1_34 = 0; ax3_0_1_34 < 2; ++ax3_0_1_34) {
    for (int ax0_68 = 0; ax0_68 < 4; ++ax0_68) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_68], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_68 * 768)) + (ax3_0_1_34 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_69 = 0; ax0_69 < 4; ++ax0_69) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_69], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_69 * 768)) + (ax3_0_1_34 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_34 = 0; ax1_0_2_34 < 4; ++ax1_0_2_34) {
      for (int ax2_0_2_34 = 0; ax2_0_2_34 < 4; ++ax2_0_2_34) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_34 * 4) + ax2_0_2_34)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_34], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_34], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_34 * 4) + ax2_0_2_34)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_70 = 0; ax0_ax1_fused_0_70 < 4; ++ax0_ax1_fused_0_70) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_70 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_70 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1120));
  }
  for (int ax0_ax1_fused_0_71 = 0; ax0_ax1_fused_0_71 < 4; ++ax0_ax1_fused_0_71) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_71 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_71 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1120));
  }
  __syncthreads();
  for (int ax3_0_1_35 = 0; ax3_0_1_35 < 2; ++ax3_0_1_35) {
    for (int ax0_70 = 0; ax0_70 < 4; ++ax0_70) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_70], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_70 * 768)) + (ax3_0_1_35 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_71 = 0; ax0_71 < 4; ++ax0_71) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_71], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_71 * 768)) + (ax3_0_1_35 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_35 = 0; ax1_0_2_35 < 4; ++ax1_0_2_35) {
      for (int ax2_0_2_35 = 0; ax2_0_2_35 < 4; ++ax2_0_2_35) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_35 * 4) + ax2_0_2_35)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_35], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_35], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_35 * 4) + ax2_0_2_35)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_72 = 0; ax0_ax1_fused_0_72 < 4; ++ax0_ax1_fused_0_72) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_72 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_72 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1152));
  }
  for (int ax0_ax1_fused_0_73 = 0; ax0_ax1_fused_0_73 < 4; ++ax0_ax1_fused_0_73) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_73 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_73 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1152));
  }
  __syncthreads();
  for (int ax3_0_1_36 = 0; ax3_0_1_36 < 2; ++ax3_0_1_36) {
    for (int ax0_72 = 0; ax0_72 < 4; ++ax0_72) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_72], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_72 * 768)) + (ax3_0_1_36 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_73 = 0; ax0_73 < 4; ++ax0_73) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_73], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_73 * 768)) + (ax3_0_1_36 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_36 = 0; ax1_0_2_36 < 4; ++ax1_0_2_36) {
      for (int ax2_0_2_36 = 0; ax2_0_2_36 < 4; ++ax2_0_2_36) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_36 * 4) + ax2_0_2_36)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_36], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_36], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_36 * 4) + ax2_0_2_36)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_74 = 0; ax0_ax1_fused_0_74 < 4; ++ax0_ax1_fused_0_74) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_74 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_74 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1184));
  }
  for (int ax0_ax1_fused_0_75 = 0; ax0_ax1_fused_0_75 < 4; ++ax0_ax1_fused_0_75) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_75 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_75 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1184));
  }
  __syncthreads();
  for (int ax3_0_1_37 = 0; ax3_0_1_37 < 2; ++ax3_0_1_37) {
    for (int ax0_74 = 0; ax0_74 < 4; ++ax0_74) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_74], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_74 * 768)) + (ax3_0_1_37 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_75 = 0; ax0_75 < 4; ++ax0_75) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_75], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_75 * 768)) + (ax3_0_1_37 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_37 = 0; ax1_0_2_37 < 4; ++ax1_0_2_37) {
      for (int ax2_0_2_37 = 0; ax2_0_2_37 < 4; ++ax2_0_2_37) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_37 * 4) + ax2_0_2_37)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_37], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_37], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_37 * 4) + ax2_0_2_37)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_76 = 0; ax0_ax1_fused_0_76 < 4; ++ax0_ax1_fused_0_76) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_76 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_76 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1216));
  }
  for (int ax0_ax1_fused_0_77 = 0; ax0_ax1_fused_0_77 < 4; ++ax0_ax1_fused_0_77) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_77 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_77 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1216));
  }
  __syncthreads();
  for (int ax3_0_1_38 = 0; ax3_0_1_38 < 2; ++ax3_0_1_38) {
    for (int ax0_76 = 0; ax0_76 < 4; ++ax0_76) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_76], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_76 * 768)) + (ax3_0_1_38 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_77 = 0; ax0_77 < 4; ++ax0_77) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_77], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_77 * 768)) + (ax3_0_1_38 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_38 = 0; ax1_0_2_38 < 4; ++ax1_0_2_38) {
      for (int ax2_0_2_38 = 0; ax2_0_2_38 < 4; ++ax2_0_2_38) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_38 * 4) + ax2_0_2_38)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_38], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_38], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_38 * 4) + ax2_0_2_38)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_78 = 0; ax0_ax1_fused_0_78 < 4; ++ax0_ax1_fused_0_78) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_78 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_78 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1248));
  }
  for (int ax0_ax1_fused_0_79 = 0; ax0_ax1_fused_0_79 < 4; ++ax0_ax1_fused_0_79) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_79 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_79 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1248));
  }
  __syncthreads();
  for (int ax3_0_1_39 = 0; ax3_0_1_39 < 2; ++ax3_0_1_39) {
    for (int ax0_78 = 0; ax0_78 < 4; ++ax0_78) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_78], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_78 * 768)) + (ax3_0_1_39 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_79 = 0; ax0_79 < 4; ++ax0_79) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_79], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_79 * 768)) + (ax3_0_1_39 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_39 = 0; ax1_0_2_39 < 4; ++ax1_0_2_39) {
      for (int ax2_0_2_39 = 0; ax2_0_2_39 < 4; ++ax2_0_2_39) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_39 * 4) + ax2_0_2_39)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_39], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_39], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_39 * 4) + ax2_0_2_39)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_80 = 0; ax0_ax1_fused_0_80 < 4; ++ax0_ax1_fused_0_80) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_80 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_80 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1280));
  }
  for (int ax0_ax1_fused_0_81 = 0; ax0_ax1_fused_0_81 < 4; ++ax0_ax1_fused_0_81) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_81 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_81 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1280));
  }
  __syncthreads();
  for (int ax3_0_1_40 = 0; ax3_0_1_40 < 2; ++ax3_0_1_40) {
    for (int ax0_80 = 0; ax0_80 < 4; ++ax0_80) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_80], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_80 * 768)) + (ax3_0_1_40 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_81 = 0; ax0_81 < 4; ++ax0_81) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_81], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_81 * 768)) + (ax3_0_1_40 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_40 = 0; ax1_0_2_40 < 4; ++ax1_0_2_40) {
      for (int ax2_0_2_40 = 0; ax2_0_2_40 < 4; ++ax2_0_2_40) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_40 * 4) + ax2_0_2_40)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_40], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_40], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_40 * 4) + ax2_0_2_40)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_82 = 0; ax0_ax1_fused_0_82 < 4; ++ax0_ax1_fused_0_82) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_82 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_82 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1312));
  }
  for (int ax0_ax1_fused_0_83 = 0; ax0_ax1_fused_0_83 < 4; ++ax0_ax1_fused_0_83) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_83 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_83 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1312));
  }
  __syncthreads();
  for (int ax3_0_1_41 = 0; ax3_0_1_41 < 2; ++ax3_0_1_41) {
    for (int ax0_82 = 0; ax0_82 < 4; ++ax0_82) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_82], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_82 * 768)) + (ax3_0_1_41 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_83 = 0; ax0_83 < 4; ++ax0_83) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_83], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_83 * 768)) + (ax3_0_1_41 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_41 = 0; ax1_0_2_41 < 4; ++ax1_0_2_41) {
      for (int ax2_0_2_41 = 0; ax2_0_2_41 < 4; ++ax2_0_2_41) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_41 * 4) + ax2_0_2_41)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_41], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_41], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_41 * 4) + ax2_0_2_41)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_84 = 0; ax0_ax1_fused_0_84 < 4; ++ax0_ax1_fused_0_84) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_84 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_84 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1344));
  }
  for (int ax0_ax1_fused_0_85 = 0; ax0_ax1_fused_0_85 < 4; ++ax0_ax1_fused_0_85) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_85 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_85 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1344));
  }
  __syncthreads();
  for (int ax3_0_1_42 = 0; ax3_0_1_42 < 2; ++ax3_0_1_42) {
    for (int ax0_84 = 0; ax0_84 < 4; ++ax0_84) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_84], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_84 * 768)) + (ax3_0_1_42 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_85 = 0; ax0_85 < 4; ++ax0_85) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_85], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_85 * 768)) + (ax3_0_1_42 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_42 = 0; ax1_0_2_42 < 4; ++ax1_0_2_42) {
      for (int ax2_0_2_42 = 0; ax2_0_2_42 < 4; ++ax2_0_2_42) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_42 * 4) + ax2_0_2_42)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_42], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_42], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_42 * 4) + ax2_0_2_42)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_86 = 0; ax0_ax1_fused_0_86 < 4; ++ax0_ax1_fused_0_86) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_86 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_86 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1376));
  }
  for (int ax0_ax1_fused_0_87 = 0; ax0_ax1_fused_0_87 < 4; ++ax0_ax1_fused_0_87) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_87 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_87 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1376));
  }
  __syncthreads();
  for (int ax3_0_1_43 = 0; ax3_0_1_43 < 2; ++ax3_0_1_43) {
    for (int ax0_86 = 0; ax0_86 < 4; ++ax0_86) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_86], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_86 * 768)) + (ax3_0_1_43 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_87 = 0; ax0_87 < 4; ++ax0_87) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_87], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_87 * 768)) + (ax3_0_1_43 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_43 = 0; ax1_0_2_43 < 4; ++ax1_0_2_43) {
      for (int ax2_0_2_43 = 0; ax2_0_2_43 < 4; ++ax2_0_2_43) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_43 * 4) + ax2_0_2_43)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_43], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_43], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_43 * 4) + ax2_0_2_43)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_88 = 0; ax0_ax1_fused_0_88 < 4; ++ax0_ax1_fused_0_88) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_88 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_88 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1408));
  }
  for (int ax0_ax1_fused_0_89 = 0; ax0_ax1_fused_0_89 < 4; ++ax0_ax1_fused_0_89) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_89 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_89 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1408));
  }
  __syncthreads();
  for (int ax3_0_1_44 = 0; ax3_0_1_44 < 2; ++ax3_0_1_44) {
    for (int ax0_88 = 0; ax0_88 < 4; ++ax0_88) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_88], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_88 * 768)) + (ax3_0_1_44 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_89 = 0; ax0_89 < 4; ++ax0_89) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_89], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_89 * 768)) + (ax3_0_1_44 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_44 = 0; ax1_0_2_44 < 4; ++ax1_0_2_44) {
      for (int ax2_0_2_44 = 0; ax2_0_2_44 < 4; ++ax2_0_2_44) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_44 * 4) + ax2_0_2_44)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_44], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_44], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_44 * 4) + ax2_0_2_44)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_90 = 0; ax0_ax1_fused_0_90 < 4; ++ax0_ax1_fused_0_90) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_90 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_90 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1440));
  }
  for (int ax0_ax1_fused_0_91 = 0; ax0_ax1_fused_0_91 < 4; ++ax0_ax1_fused_0_91) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_91 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_91 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1440));
  }
  __syncthreads();
  for (int ax3_0_1_45 = 0; ax3_0_1_45 < 2; ++ax3_0_1_45) {
    for (int ax0_90 = 0; ax0_90 < 4; ++ax0_90) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_90], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_90 * 768)) + (ax3_0_1_45 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_91 = 0; ax0_91 < 4; ++ax0_91) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_91], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_91 * 768)) + (ax3_0_1_45 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_45 = 0; ax1_0_2_45 < 4; ++ax1_0_2_45) {
      for (int ax2_0_2_45 = 0; ax2_0_2_45 < 4; ++ax2_0_2_45) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_45 * 4) + ax2_0_2_45)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_45], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_45], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_45 * 4) + ax2_0_2_45)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_92 = 0; ax0_ax1_fused_0_92 < 4; ++ax0_ax1_fused_0_92) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_92 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_92 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1472));
  }
  for (int ax0_ax1_fused_0_93 = 0; ax0_ax1_fused_0_93 < 4; ++ax0_ax1_fused_0_93) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_93 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_93 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1472));
  }
  __syncthreads();
  for (int ax3_0_1_46 = 0; ax3_0_1_46 < 2; ++ax3_0_1_46) {
    for (int ax0_92 = 0; ax0_92 < 4; ++ax0_92) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_92], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_92 * 768)) + (ax3_0_1_46 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_93 = 0; ax0_93 < 4; ++ax0_93) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_93], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_93 * 768)) + (ax3_0_1_46 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_46 = 0; ax1_0_2_46 < 4; ++ax1_0_2_46) {
      for (int ax2_0_2_46 = 0; ax2_0_2_46 < 4; ++ax2_0_2_46) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_46 * 4) + ax2_0_2_46)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_46], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_46], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_46 * 4) + ax2_0_2_46)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_94 = 0; ax0_ax1_fused_0_94 < 4; ++ax0_ax1_fused_0_94) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_94 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_94 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1504));
  }
  for (int ax0_ax1_fused_0_95 = 0; ax0_ax1_fused_0_95 < 4; ++ax0_ax1_fused_0_95) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_95 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_95 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1504));
  }
  __syncthreads();
  for (int ax3_0_1_47 = 0; ax3_0_1_47 < 2; ++ax3_0_1_47) {
    for (int ax0_94 = 0; ax0_94 < 4; ++ax0_94) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_94], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_94 * 768)) + (ax3_0_1_47 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_95 = 0; ax0_95 < 4; ++ax0_95) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_95], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_95 * 768)) + (ax3_0_1_47 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_47 = 0; ax1_0_2_47 < 4; ++ax1_0_2_47) {
      for (int ax2_0_2_47 = 0; ax2_0_2_47 < 4; ++ax2_0_2_47) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_47 * 4) + ax2_0_2_47)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_47], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_47], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_47 * 4) + ax2_0_2_47)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_96 = 0; ax0_ax1_fused_0_96 < 4; ++ax0_ax1_fused_0_96) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_96 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_96 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1536));
  }
  for (int ax0_ax1_fused_0_97 = 0; ax0_ax1_fused_0_97 < 4; ++ax0_ax1_fused_0_97) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_97 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_97 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1536));
  }
  __syncthreads();
  for (int ax3_0_1_48 = 0; ax3_0_1_48 < 2; ++ax3_0_1_48) {
    for (int ax0_96 = 0; ax0_96 < 4; ++ax0_96) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_96], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_96 * 768)) + (ax3_0_1_48 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_97 = 0; ax0_97 < 4; ++ax0_97) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_97], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_97 * 768)) + (ax3_0_1_48 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_48 = 0; ax1_0_2_48 < 4; ++ax1_0_2_48) {
      for (int ax2_0_2_48 = 0; ax2_0_2_48 < 4; ++ax2_0_2_48) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_48 * 4) + ax2_0_2_48)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_48], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_48], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_48 * 4) + ax2_0_2_48)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_98 = 0; ax0_ax1_fused_0_98 < 4; ++ax0_ax1_fused_0_98) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_98 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_98 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1568));
  }
  for (int ax0_ax1_fused_0_99 = 0; ax0_ax1_fused_0_99 < 4; ++ax0_ax1_fused_0_99) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_99 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_99 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1568));
  }
  __syncthreads();
  for (int ax3_0_1_49 = 0; ax3_0_1_49 < 2; ++ax3_0_1_49) {
    for (int ax0_98 = 0; ax0_98 < 4; ++ax0_98) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_98], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_98 * 768)) + (ax3_0_1_49 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_99 = 0; ax0_99 < 4; ++ax0_99) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_99], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_99 * 768)) + (ax3_0_1_49 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_49 = 0; ax1_0_2_49 < 4; ++ax1_0_2_49) {
      for (int ax2_0_2_49 = 0; ax2_0_2_49 < 4; ++ax2_0_2_49) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_49 * 4) + ax2_0_2_49)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_49], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_49], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_49 * 4) + ax2_0_2_49)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_100 = 0; ax0_ax1_fused_0_100 < 4; ++ax0_ax1_fused_0_100) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_100 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_100 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1600));
  }
  for (int ax0_ax1_fused_0_101 = 0; ax0_ax1_fused_0_101 < 4; ++ax0_ax1_fused_0_101) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_101 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_101 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1600));
  }
  __syncthreads();
  for (int ax3_0_1_50 = 0; ax3_0_1_50 < 2; ++ax3_0_1_50) {
    for (int ax0_100 = 0; ax0_100 < 4; ++ax0_100) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_100], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_100 * 768)) + (ax3_0_1_50 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_101 = 0; ax0_101 < 4; ++ax0_101) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_101], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_101 * 768)) + (ax3_0_1_50 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_50 = 0; ax1_0_2_50 < 4; ++ax1_0_2_50) {
      for (int ax2_0_2_50 = 0; ax2_0_2_50 < 4; ++ax2_0_2_50) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_50 * 4) + ax2_0_2_50)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_50], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_50], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_50 * 4) + ax2_0_2_50)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_102 = 0; ax0_ax1_fused_0_102 < 4; ++ax0_ax1_fused_0_102) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_102 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_102 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1632));
  }
  for (int ax0_ax1_fused_0_103 = 0; ax0_ax1_fused_0_103 < 4; ++ax0_ax1_fused_0_103) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_103 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_103 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1632));
  }
  __syncthreads();
  for (int ax3_0_1_51 = 0; ax3_0_1_51 < 2; ++ax3_0_1_51) {
    for (int ax0_102 = 0; ax0_102 < 4; ++ax0_102) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_102], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_102 * 768)) + (ax3_0_1_51 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_103 = 0; ax0_103 < 4; ++ax0_103) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_103], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_103 * 768)) + (ax3_0_1_51 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_51 = 0; ax1_0_2_51 < 4; ++ax1_0_2_51) {
      for (int ax2_0_2_51 = 0; ax2_0_2_51 < 4; ++ax2_0_2_51) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_51 * 4) + ax2_0_2_51)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_51], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_51], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_51 * 4) + ax2_0_2_51)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_104 = 0; ax0_ax1_fused_0_104 < 4; ++ax0_ax1_fused_0_104) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_104 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_104 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1664));
  }
  for (int ax0_ax1_fused_0_105 = 0; ax0_ax1_fused_0_105 < 4; ++ax0_ax1_fused_0_105) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_105 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_105 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1664));
  }
  __syncthreads();
  for (int ax3_0_1_52 = 0; ax3_0_1_52 < 2; ++ax3_0_1_52) {
    for (int ax0_104 = 0; ax0_104 < 4; ++ax0_104) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_104], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_104 * 768)) + (ax3_0_1_52 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_105 = 0; ax0_105 < 4; ++ax0_105) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_105], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_105 * 768)) + (ax3_0_1_52 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_52 = 0; ax1_0_2_52 < 4; ++ax1_0_2_52) {
      for (int ax2_0_2_52 = 0; ax2_0_2_52 < 4; ++ax2_0_2_52) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_52 * 4) + ax2_0_2_52)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_52], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_52], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_52 * 4) + ax2_0_2_52)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_106 = 0; ax0_ax1_fused_0_106 < 4; ++ax0_ax1_fused_0_106) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_106 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_106 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1696));
  }
  for (int ax0_ax1_fused_0_107 = 0; ax0_ax1_fused_0_107 < 4; ++ax0_ax1_fused_0_107) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_107 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_107 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1696));
  }
  __syncthreads();
  for (int ax3_0_1_53 = 0; ax3_0_1_53 < 2; ++ax3_0_1_53) {
    for (int ax0_106 = 0; ax0_106 < 4; ++ax0_106) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_106], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_106 * 768)) + (ax3_0_1_53 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_107 = 0; ax0_107 < 4; ++ax0_107) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_107], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_107 * 768)) + (ax3_0_1_53 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_53 = 0; ax1_0_2_53 < 4; ++ax1_0_2_53) {
      for (int ax2_0_2_53 = 0; ax2_0_2_53 < 4; ++ax2_0_2_53) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_53 * 4) + ax2_0_2_53)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_53], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_53], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_53 * 4) + ax2_0_2_53)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_108 = 0; ax0_ax1_fused_0_108 < 4; ++ax0_ax1_fused_0_108) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_108 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_108 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1728));
  }
  for (int ax0_ax1_fused_0_109 = 0; ax0_ax1_fused_0_109 < 4; ++ax0_ax1_fused_0_109) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_109 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_109 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1728));
  }
  __syncthreads();
  for (int ax3_0_1_54 = 0; ax3_0_1_54 < 2; ++ax3_0_1_54) {
    for (int ax0_108 = 0; ax0_108 < 4; ++ax0_108) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_108], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_108 * 768)) + (ax3_0_1_54 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_109 = 0; ax0_109 < 4; ++ax0_109) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_109], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_109 * 768)) + (ax3_0_1_54 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_54 = 0; ax1_0_2_54 < 4; ++ax1_0_2_54) {
      for (int ax2_0_2_54 = 0; ax2_0_2_54 < 4; ++ax2_0_2_54) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_54 * 4) + ax2_0_2_54)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_54], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_54], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_54 * 4) + ax2_0_2_54)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_110 = 0; ax0_ax1_fused_0_110 < 4; ++ax0_ax1_fused_0_110) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_110 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_110 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1760));
  }
  for (int ax0_ax1_fused_0_111 = 0; ax0_ax1_fused_0_111 < 4; ++ax0_ax1_fused_0_111) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_111 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_111 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1760));
  }
  __syncthreads();
  for (int ax3_0_1_55 = 0; ax3_0_1_55 < 2; ++ax3_0_1_55) {
    for (int ax0_110 = 0; ax0_110 < 4; ++ax0_110) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_110], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_110 * 768)) + (ax3_0_1_55 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_111 = 0; ax0_111 < 4; ++ax0_111) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_111], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_111 * 768)) + (ax3_0_1_55 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_55 = 0; ax1_0_2_55 < 4; ++ax1_0_2_55) {
      for (int ax2_0_2_55 = 0; ax2_0_2_55 < 4; ++ax2_0_2_55) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_55 * 4) + ax2_0_2_55)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_55], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_55], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_55 * 4) + ax2_0_2_55)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_112 = 0; ax0_ax1_fused_0_112 < 4; ++ax0_ax1_fused_0_112) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_112 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_112 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1792));
  }
  for (int ax0_ax1_fused_0_113 = 0; ax0_ax1_fused_0_113 < 4; ++ax0_ax1_fused_0_113) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_113 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_113 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1792));
  }
  __syncthreads();
  for (int ax3_0_1_56 = 0; ax3_0_1_56 < 2; ++ax3_0_1_56) {
    for (int ax0_112 = 0; ax0_112 < 4; ++ax0_112) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_112], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_112 * 768)) + (ax3_0_1_56 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_113 = 0; ax0_113 < 4; ++ax0_113) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_113], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_113 * 768)) + (ax3_0_1_56 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_56 = 0; ax1_0_2_56 < 4; ++ax1_0_2_56) {
      for (int ax2_0_2_56 = 0; ax2_0_2_56 < 4; ++ax2_0_2_56) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_56 * 4) + ax2_0_2_56)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_56], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_56], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_56 * 4) + ax2_0_2_56)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_114 = 0; ax0_ax1_fused_0_114 < 4; ++ax0_ax1_fused_0_114) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_114 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_114 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1824));
  }
  for (int ax0_ax1_fused_0_115 = 0; ax0_ax1_fused_0_115 < 4; ++ax0_ax1_fused_0_115) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_115 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_115 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1824));
  }
  __syncthreads();
  for (int ax3_0_1_57 = 0; ax3_0_1_57 < 2; ++ax3_0_1_57) {
    for (int ax0_114 = 0; ax0_114 < 4; ++ax0_114) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_114], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_114 * 768)) + (ax3_0_1_57 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_115 = 0; ax0_115 < 4; ++ax0_115) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_115], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_115 * 768)) + (ax3_0_1_57 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_57 = 0; ax1_0_2_57 < 4; ++ax1_0_2_57) {
      for (int ax2_0_2_57 = 0; ax2_0_2_57 < 4; ++ax2_0_2_57) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_57 * 4) + ax2_0_2_57)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_57], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_57], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_57 * 4) + ax2_0_2_57)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_116 = 0; ax0_ax1_fused_0_116 < 4; ++ax0_ax1_fused_0_116) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_116 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_116 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1856));
  }
  for (int ax0_ax1_fused_0_117 = 0; ax0_ax1_fused_0_117 < 4; ++ax0_ax1_fused_0_117) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_117 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_117 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1856));
  }
  __syncthreads();
  for (int ax3_0_1_58 = 0; ax3_0_1_58 < 2; ++ax3_0_1_58) {
    for (int ax0_116 = 0; ax0_116 < 4; ++ax0_116) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_116], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_116 * 768)) + (ax3_0_1_58 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_117 = 0; ax0_117 < 4; ++ax0_117) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_117], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_117 * 768)) + (ax3_0_1_58 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_58 = 0; ax1_0_2_58 < 4; ++ax1_0_2_58) {
      for (int ax2_0_2_58 = 0; ax2_0_2_58 < 4; ++ax2_0_2_58) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_58 * 4) + ax2_0_2_58)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_58], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_58], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_58 * 4) + ax2_0_2_58)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_118 = 0; ax0_ax1_fused_0_118 < 4; ++ax0_ax1_fused_0_118) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_118 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_118 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1888));
  }
  for (int ax0_ax1_fused_0_119 = 0; ax0_ax1_fused_0_119 < 4; ++ax0_ax1_fused_0_119) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_119 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_119 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1888));
  }
  __syncthreads();
  for (int ax3_0_1_59 = 0; ax3_0_1_59 < 2; ++ax3_0_1_59) {
    for (int ax0_118 = 0; ax0_118 < 4; ++ax0_118) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_118], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_118 * 768)) + (ax3_0_1_59 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_119 = 0; ax0_119 < 4; ++ax0_119) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_119], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_119 * 768)) + (ax3_0_1_59 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_59 = 0; ax1_0_2_59 < 4; ++ax1_0_2_59) {
      for (int ax2_0_2_59 = 0; ax2_0_2_59 < 4; ++ax2_0_2_59) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_59 * 4) + ax2_0_2_59)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_59], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_59], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_59 * 4) + ax2_0_2_59)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_120 = 0; ax0_ax1_fused_0_120 < 4; ++ax0_ax1_fused_0_120) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_120 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_120 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1920));
  }
  for (int ax0_ax1_fused_0_121 = 0; ax0_ax1_fused_0_121 < 4; ++ax0_ax1_fused_0_121) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_121 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_121 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1920));
  }
  __syncthreads();
  for (int ax3_0_1_60 = 0; ax3_0_1_60 < 2; ++ax3_0_1_60) {
    for (int ax0_120 = 0; ax0_120 < 4; ++ax0_120) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_120], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_120 * 768)) + (ax3_0_1_60 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_121 = 0; ax0_121 < 4; ++ax0_121) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_121], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_121 * 768)) + (ax3_0_1_60 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_60 = 0; ax1_0_2_60 < 4; ++ax1_0_2_60) {
      for (int ax2_0_2_60 = 0; ax2_0_2_60 < 4; ++ax2_0_2_60) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_60 * 4) + ax2_0_2_60)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_60], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_60], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_60 * 4) + ax2_0_2_60)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_122 = 0; ax0_ax1_fused_0_122 < 4; ++ax0_ax1_fused_0_122) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_122 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_122 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1952));
  }
  for (int ax0_ax1_fused_0_123 = 0; ax0_ax1_fused_0_123 < 4; ++ax0_ax1_fused_0_123) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_123 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_123 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1952));
  }
  __syncthreads();
  for (int ax3_0_1_61 = 0; ax3_0_1_61 < 2; ++ax3_0_1_61) {
    for (int ax0_122 = 0; ax0_122 < 4; ++ax0_122) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_122], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_122 * 768)) + (ax3_0_1_61 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_123 = 0; ax0_123 < 4; ++ax0_123) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_123], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_123 * 768)) + (ax3_0_1_61 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_61 = 0; ax1_0_2_61 < 4; ++ax1_0_2_61) {
      for (int ax2_0_2_61 = 0; ax2_0_2_61 < 4; ++ax2_0_2_61) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_61 * 4) + ax2_0_2_61)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_61], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_61], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_61 * 4) + ax2_0_2_61)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_124 = 0; ax0_ax1_fused_0_124 < 4; ++ax0_ax1_fused_0_124) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_124 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_124 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1984));
  }
  for (int ax0_ax1_fused_0_125 = 0; ax0_ax1_fused_0_125 < 4; ++ax0_ax1_fused_0_125) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_125 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_125 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1984));
  }
  __syncthreads();
  for (int ax3_0_1_62 = 0; ax3_0_1_62 < 2; ++ax3_0_1_62) {
    for (int ax0_124 = 0; ax0_124 < 4; ++ax0_124) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_124], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_124 * 768)) + (ax3_0_1_62 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_125 = 0; ax0_125 < 4; ++ax0_125) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_125], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_125 * 768)) + (ax3_0_1_62 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_62 = 0; ax1_0_2_62 < 4; ++ax1_0_2_62) {
      for (int ax2_0_2_62 = 0; ax2_0_2_62 < 4; ++ax2_0_2_62) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_62 * 4) + ax2_0_2_62)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_62], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_62], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_62 * 4) + ax2_0_2_62)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_126 = 0; ax0_ax1_fused_0_126 < 4; ++ax0_ax1_fused_0_126) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_126 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_126 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2016));
  }
  for (int ax0_ax1_fused_0_127 = 0; ax0_ax1_fused_0_127 < 4; ++ax0_ax1_fused_0_127) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_127 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_127 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2016));
  }
  __syncthreads();
  for (int ax3_0_1_63 = 0; ax3_0_1_63 < 2; ++ax3_0_1_63) {
    for (int ax0_126 = 0; ax0_126 < 4; ++ax0_126) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_126], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_126 * 768)) + (ax3_0_1_63 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_127 = 0; ax0_127 < 4; ++ax0_127) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_127], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_127 * 768)) + (ax3_0_1_63 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_63 = 0; ax1_0_2_63 < 4; ++ax1_0_2_63) {
      for (int ax2_0_2_63 = 0; ax2_0_2_63 < 4; ++ax2_0_2_63) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_63 * 4) + ax2_0_2_63)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_63], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_63], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_63 * 4) + ax2_0_2_63)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_128 = 0; ax0_ax1_fused_0_128 < 4; ++ax0_ax1_fused_0_128) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_128 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_128 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2048));
  }
  for (int ax0_ax1_fused_0_129 = 0; ax0_ax1_fused_0_129 < 4; ++ax0_ax1_fused_0_129) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_129 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_129 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2048));
  }
  __syncthreads();
  for (int ax3_0_1_64 = 0; ax3_0_1_64 < 2; ++ax3_0_1_64) {
    for (int ax0_128 = 0; ax0_128 < 4; ++ax0_128) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_128], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_128 * 768)) + (ax3_0_1_64 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_129 = 0; ax0_129 < 4; ++ax0_129) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_129], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_129 * 768)) + (ax3_0_1_64 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_64 = 0; ax1_0_2_64 < 4; ++ax1_0_2_64) {
      for (int ax2_0_2_64 = 0; ax2_0_2_64 < 4; ++ax2_0_2_64) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_64 * 4) + ax2_0_2_64)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_64], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_64], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_64 * 4) + ax2_0_2_64)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_130 = 0; ax0_ax1_fused_0_130 < 4; ++ax0_ax1_fused_0_130) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_130 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_130 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2080));
  }
  for (int ax0_ax1_fused_0_131 = 0; ax0_ax1_fused_0_131 < 4; ++ax0_ax1_fused_0_131) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_131 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_131 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2080));
  }
  __syncthreads();
  for (int ax3_0_1_65 = 0; ax3_0_1_65 < 2; ++ax3_0_1_65) {
    for (int ax0_130 = 0; ax0_130 < 4; ++ax0_130) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_130], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_130 * 768)) + (ax3_0_1_65 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_131 = 0; ax0_131 < 4; ++ax0_131) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_131], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_131 * 768)) + (ax3_0_1_65 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_65 = 0; ax1_0_2_65 < 4; ++ax1_0_2_65) {
      for (int ax2_0_2_65 = 0; ax2_0_2_65 < 4; ++ax2_0_2_65) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_65 * 4) + ax2_0_2_65)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_65], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_65], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_65 * 4) + ax2_0_2_65)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_132 = 0; ax0_ax1_fused_0_132 < 4; ++ax0_ax1_fused_0_132) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_132 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_132 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2112));
  }
  for (int ax0_ax1_fused_0_133 = 0; ax0_ax1_fused_0_133 < 4; ++ax0_ax1_fused_0_133) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_133 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_133 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2112));
  }
  __syncthreads();
  for (int ax3_0_1_66 = 0; ax3_0_1_66 < 2; ++ax3_0_1_66) {
    for (int ax0_132 = 0; ax0_132 < 4; ++ax0_132) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_132], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_132 * 768)) + (ax3_0_1_66 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_133 = 0; ax0_133 < 4; ++ax0_133) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_133], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_133 * 768)) + (ax3_0_1_66 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_66 = 0; ax1_0_2_66 < 4; ++ax1_0_2_66) {
      for (int ax2_0_2_66 = 0; ax2_0_2_66 < 4; ++ax2_0_2_66) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_66 * 4) + ax2_0_2_66)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_66], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_66], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_66 * 4) + ax2_0_2_66)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_134 = 0; ax0_ax1_fused_0_134 < 4; ++ax0_ax1_fused_0_134) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_134 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_134 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2144));
  }
  for (int ax0_ax1_fused_0_135 = 0; ax0_ax1_fused_0_135 < 4; ++ax0_ax1_fused_0_135) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_135 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_135 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2144));
  }
  __syncthreads();
  for (int ax3_0_1_67 = 0; ax3_0_1_67 < 2; ++ax3_0_1_67) {
    for (int ax0_134 = 0; ax0_134 < 4; ++ax0_134) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_134], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_134 * 768)) + (ax3_0_1_67 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_135 = 0; ax0_135 < 4; ++ax0_135) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_135], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_135 * 768)) + (ax3_0_1_67 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_67 = 0; ax1_0_2_67 < 4; ++ax1_0_2_67) {
      for (int ax2_0_2_67 = 0; ax2_0_2_67 < 4; ++ax2_0_2_67) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_67 * 4) + ax2_0_2_67)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_67], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_67], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_67 * 4) + ax2_0_2_67)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_136 = 0; ax0_ax1_fused_0_136 < 4; ++ax0_ax1_fused_0_136) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_136 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_136 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2176));
  }
  for (int ax0_ax1_fused_0_137 = 0; ax0_ax1_fused_0_137 < 4; ++ax0_ax1_fused_0_137) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_137 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_137 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2176));
  }
  __syncthreads();
  for (int ax3_0_1_68 = 0; ax3_0_1_68 < 2; ++ax3_0_1_68) {
    for (int ax0_136 = 0; ax0_136 < 4; ++ax0_136) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_136], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_136 * 768)) + (ax3_0_1_68 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_137 = 0; ax0_137 < 4; ++ax0_137) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_137], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_137 * 768)) + (ax3_0_1_68 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_68 = 0; ax1_0_2_68 < 4; ++ax1_0_2_68) {
      for (int ax2_0_2_68 = 0; ax2_0_2_68 < 4; ++ax2_0_2_68) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_68 * 4) + ax2_0_2_68)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_68], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_68], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_68 * 4) + ax2_0_2_68)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_138 = 0; ax0_ax1_fused_0_138 < 4; ++ax0_ax1_fused_0_138) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_138 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_138 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2208));
  }
  for (int ax0_ax1_fused_0_139 = 0; ax0_ax1_fused_0_139 < 4; ++ax0_ax1_fused_0_139) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_139 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_139 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2208));
  }
  __syncthreads();
  for (int ax3_0_1_69 = 0; ax3_0_1_69 < 2; ++ax3_0_1_69) {
    for (int ax0_138 = 0; ax0_138 < 4; ++ax0_138) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_138], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_138 * 768)) + (ax3_0_1_69 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_139 = 0; ax0_139 < 4; ++ax0_139) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_139], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_139 * 768)) + (ax3_0_1_69 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_69 = 0; ax1_0_2_69 < 4; ++ax1_0_2_69) {
      for (int ax2_0_2_69 = 0; ax2_0_2_69 < 4; ++ax2_0_2_69) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_69 * 4) + ax2_0_2_69)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_69], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_69], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_69 * 4) + ax2_0_2_69)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_140 = 0; ax0_ax1_fused_0_140 < 4; ++ax0_ax1_fused_0_140) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_140 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_140 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2240));
  }
  for (int ax0_ax1_fused_0_141 = 0; ax0_ax1_fused_0_141 < 4; ++ax0_ax1_fused_0_141) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_141 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_141 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2240));
  }
  __syncthreads();
  for (int ax3_0_1_70 = 0; ax3_0_1_70 < 2; ++ax3_0_1_70) {
    for (int ax0_140 = 0; ax0_140 < 4; ++ax0_140) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_140], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_140 * 768)) + (ax3_0_1_70 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_141 = 0; ax0_141 < 4; ++ax0_141) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_141], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_141 * 768)) + (ax3_0_1_70 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_70 = 0; ax1_0_2_70 < 4; ++ax1_0_2_70) {
      for (int ax2_0_2_70 = 0; ax2_0_2_70 < 4; ++ax2_0_2_70) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_70 * 4) + ax2_0_2_70)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_70], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_70], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_70 * 4) + ax2_0_2_70)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_142 = 0; ax0_ax1_fused_0_142 < 4; ++ax0_ax1_fused_0_142) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_142 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_142 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2272));
  }
  for (int ax0_ax1_fused_0_143 = 0; ax0_ax1_fused_0_143 < 4; ++ax0_ax1_fused_0_143) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_143 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_143 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2272));
  }
  __syncthreads();
  for (int ax3_0_1_71 = 0; ax3_0_1_71 < 2; ++ax3_0_1_71) {
    for (int ax0_142 = 0; ax0_142 < 4; ++ax0_142) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_142], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_142 * 768)) + (ax3_0_1_71 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_143 = 0; ax0_143 < 4; ++ax0_143) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_143], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_143 * 768)) + (ax3_0_1_71 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_71 = 0; ax1_0_2_71 < 4; ++ax1_0_2_71) {
      for (int ax2_0_2_71 = 0; ax2_0_2_71 < 4; ++ax2_0_2_71) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_71 * 4) + ax2_0_2_71)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_71], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_71], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_71 * 4) + ax2_0_2_71)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_144 = 0; ax0_ax1_fused_0_144 < 4; ++ax0_ax1_fused_0_144) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_144 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_144 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2304));
  }
  for (int ax0_ax1_fused_0_145 = 0; ax0_ax1_fused_0_145 < 4; ++ax0_ax1_fused_0_145) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_145 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_145 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2304));
  }
  __syncthreads();
  for (int ax3_0_1_72 = 0; ax3_0_1_72 < 2; ++ax3_0_1_72) {
    for (int ax0_144 = 0; ax0_144 < 4; ++ax0_144) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_144], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_144 * 768)) + (ax3_0_1_72 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_145 = 0; ax0_145 < 4; ++ax0_145) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_145], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_145 * 768)) + (ax3_0_1_72 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_72 = 0; ax1_0_2_72 < 4; ++ax1_0_2_72) {
      for (int ax2_0_2_72 = 0; ax2_0_2_72 < 4; ++ax2_0_2_72) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_72 * 4) + ax2_0_2_72)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_72], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_72], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_72 * 4) + ax2_0_2_72)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_146 = 0; ax0_ax1_fused_0_146 < 4; ++ax0_ax1_fused_0_146) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_146 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_146 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2336));
  }
  for (int ax0_ax1_fused_0_147 = 0; ax0_ax1_fused_0_147 < 4; ++ax0_ax1_fused_0_147) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_147 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_147 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2336));
  }
  __syncthreads();
  for (int ax3_0_1_73 = 0; ax3_0_1_73 < 2; ++ax3_0_1_73) {
    for (int ax0_146 = 0; ax0_146 < 4; ++ax0_146) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_146], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_146 * 768)) + (ax3_0_1_73 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_147 = 0; ax0_147 < 4; ++ax0_147) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_147], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_147 * 768)) + (ax3_0_1_73 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_73 = 0; ax1_0_2_73 < 4; ++ax1_0_2_73) {
      for (int ax2_0_2_73 = 0; ax2_0_2_73 < 4; ++ax2_0_2_73) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_73 * 4) + ax2_0_2_73)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_73], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_73], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_73 * 4) + ax2_0_2_73)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_148 = 0; ax0_ax1_fused_0_148 < 4; ++ax0_ax1_fused_0_148) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_148 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_148 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2368));
  }
  for (int ax0_ax1_fused_0_149 = 0; ax0_ax1_fused_0_149 < 4; ++ax0_ax1_fused_0_149) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_149 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_149 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2368));
  }
  __syncthreads();
  for (int ax3_0_1_74 = 0; ax3_0_1_74 < 2; ++ax3_0_1_74) {
    for (int ax0_148 = 0; ax0_148 < 4; ++ax0_148) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_148], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_148 * 768)) + (ax3_0_1_74 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_149 = 0; ax0_149 < 4; ++ax0_149) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_149], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_149 * 768)) + (ax3_0_1_74 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_74 = 0; ax1_0_2_74 < 4; ++ax1_0_2_74) {
      for (int ax2_0_2_74 = 0; ax2_0_2_74 < 4; ++ax2_0_2_74) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_74 * 4) + ax2_0_2_74)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_74], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_74], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_74 * 4) + ax2_0_2_74)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_150 = 0; ax0_ax1_fused_0_150 < 4; ++ax0_ax1_fused_0_150) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_150 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_150 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2400));
  }
  for (int ax0_ax1_fused_0_151 = 0; ax0_ax1_fused_0_151 < 4; ++ax0_ax1_fused_0_151) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_151 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_151 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2400));
  }
  __syncthreads();
  for (int ax3_0_1_75 = 0; ax3_0_1_75 < 2; ++ax3_0_1_75) {
    for (int ax0_150 = 0; ax0_150 < 4; ++ax0_150) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_150], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_150 * 768)) + (ax3_0_1_75 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_151 = 0; ax0_151 < 4; ++ax0_151) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_151], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_151 * 768)) + (ax3_0_1_75 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_75 = 0; ax1_0_2_75 < 4; ++ax1_0_2_75) {
      for (int ax2_0_2_75 = 0; ax2_0_2_75 < 4; ++ax2_0_2_75) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_75 * 4) + ax2_0_2_75)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_75], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_75], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_75 * 4) + ax2_0_2_75)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_152 = 0; ax0_ax1_fused_0_152 < 4; ++ax0_ax1_fused_0_152) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_152 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_152 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2432));
  }
  for (int ax0_ax1_fused_0_153 = 0; ax0_ax1_fused_0_153 < 4; ++ax0_ax1_fused_0_153) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_153 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_153 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2432));
  }
  __syncthreads();
  for (int ax3_0_1_76 = 0; ax3_0_1_76 < 2; ++ax3_0_1_76) {
    for (int ax0_152 = 0; ax0_152 < 4; ++ax0_152) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_152], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_152 * 768)) + (ax3_0_1_76 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_153 = 0; ax0_153 < 4; ++ax0_153) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_153], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_153 * 768)) + (ax3_0_1_76 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_76 = 0; ax1_0_2_76 < 4; ++ax1_0_2_76) {
      for (int ax2_0_2_76 = 0; ax2_0_2_76 < 4; ++ax2_0_2_76) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_76 * 4) + ax2_0_2_76)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_76], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_76], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_76 * 4) + ax2_0_2_76)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_154 = 0; ax0_ax1_fused_0_154 < 4; ++ax0_ax1_fused_0_154) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_154 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_154 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2464));
  }
  for (int ax0_ax1_fused_0_155 = 0; ax0_ax1_fused_0_155 < 4; ++ax0_ax1_fused_0_155) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_155 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_155 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2464));
  }
  __syncthreads();
  for (int ax3_0_1_77 = 0; ax3_0_1_77 < 2; ++ax3_0_1_77) {
    for (int ax0_154 = 0; ax0_154 < 4; ++ax0_154) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_154], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_154 * 768)) + (ax3_0_1_77 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_155 = 0; ax0_155 < 4; ++ax0_155) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_155], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_155 * 768)) + (ax3_0_1_77 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_77 = 0; ax1_0_2_77 < 4; ++ax1_0_2_77) {
      for (int ax2_0_2_77 = 0; ax2_0_2_77 < 4; ++ax2_0_2_77) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_77 * 4) + ax2_0_2_77)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_77], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_77], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_77 * 4) + ax2_0_2_77)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_156 = 0; ax0_ax1_fused_0_156 < 4; ++ax0_ax1_fused_0_156) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_156 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_156 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2496));
  }
  for (int ax0_ax1_fused_0_157 = 0; ax0_ax1_fused_0_157 < 4; ++ax0_ax1_fused_0_157) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_157 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_157 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2496));
  }
  __syncthreads();
  for (int ax3_0_1_78 = 0; ax3_0_1_78 < 2; ++ax3_0_1_78) {
    for (int ax0_156 = 0; ax0_156 < 4; ++ax0_156) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_156], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_156 * 768)) + (ax3_0_1_78 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_157 = 0; ax0_157 < 4; ++ax0_157) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_157], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_157 * 768)) + (ax3_0_1_78 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_78 = 0; ax1_0_2_78 < 4; ++ax1_0_2_78) {
      for (int ax2_0_2_78 = 0; ax2_0_2_78 < 4; ++ax2_0_2_78) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_78 * 4) + ax2_0_2_78)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_78], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_78], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_78 * 4) + ax2_0_2_78)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_158 = 0; ax0_ax1_fused_0_158 < 4; ++ax0_ax1_fused_0_158) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_158 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_158 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2528));
  }
  for (int ax0_ax1_fused_0_159 = 0; ax0_ax1_fused_0_159 < 4; ++ax0_ax1_fused_0_159) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_159 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_159 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2528));
  }
  __syncthreads();
  for (int ax3_0_1_79 = 0; ax3_0_1_79 < 2; ++ax3_0_1_79) {
    for (int ax0_158 = 0; ax0_158 < 4; ++ax0_158) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_158], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_158 * 768)) + (ax3_0_1_79 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_159 = 0; ax0_159 < 4; ++ax0_159) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_159], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_159 * 768)) + (ax3_0_1_79 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_79 = 0; ax1_0_2_79 < 4; ++ax1_0_2_79) {
      for (int ax2_0_2_79 = 0; ax2_0_2_79 < 4; ++ax2_0_2_79) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_79 * 4) + ax2_0_2_79)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_79], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_79], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_79 * 4) + ax2_0_2_79)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_160 = 0; ax0_ax1_fused_0_160 < 4; ++ax0_ax1_fused_0_160) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_160 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_160 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2560));
  }
  for (int ax0_ax1_fused_0_161 = 0; ax0_ax1_fused_0_161 < 4; ++ax0_ax1_fused_0_161) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_161 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_161 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2560));
  }
  __syncthreads();
  for (int ax3_0_1_80 = 0; ax3_0_1_80 < 2; ++ax3_0_1_80) {
    for (int ax0_160 = 0; ax0_160 < 4; ++ax0_160) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_160], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_160 * 768)) + (ax3_0_1_80 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_161 = 0; ax0_161 < 4; ++ax0_161) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_161], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_161 * 768)) + (ax3_0_1_80 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_80 = 0; ax1_0_2_80 < 4; ++ax1_0_2_80) {
      for (int ax2_0_2_80 = 0; ax2_0_2_80 < 4; ++ax2_0_2_80) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_80 * 4) + ax2_0_2_80)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_80], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_80], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_80 * 4) + ax2_0_2_80)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_162 = 0; ax0_ax1_fused_0_162 < 4; ++ax0_ax1_fused_0_162) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_162 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_162 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2592));
  }
  for (int ax0_ax1_fused_0_163 = 0; ax0_ax1_fused_0_163 < 4; ++ax0_ax1_fused_0_163) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_163 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_163 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2592));
  }
  __syncthreads();
  for (int ax3_0_1_81 = 0; ax3_0_1_81 < 2; ++ax3_0_1_81) {
    for (int ax0_162 = 0; ax0_162 < 4; ++ax0_162) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_162], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_162 * 768)) + (ax3_0_1_81 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_163 = 0; ax0_163 < 4; ++ax0_163) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_163], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_163 * 768)) + (ax3_0_1_81 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_81 = 0; ax1_0_2_81 < 4; ++ax1_0_2_81) {
      for (int ax2_0_2_81 = 0; ax2_0_2_81 < 4; ++ax2_0_2_81) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_81 * 4) + ax2_0_2_81)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_81], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_81], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_81 * 4) + ax2_0_2_81)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_164 = 0; ax0_ax1_fused_0_164 < 4; ++ax0_ax1_fused_0_164) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_164 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_164 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2624));
  }
  for (int ax0_ax1_fused_0_165 = 0; ax0_ax1_fused_0_165 < 4; ++ax0_ax1_fused_0_165) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_165 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_165 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2624));
  }
  __syncthreads();
  for (int ax3_0_1_82 = 0; ax3_0_1_82 < 2; ++ax3_0_1_82) {
    for (int ax0_164 = 0; ax0_164 < 4; ++ax0_164) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_164], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_164 * 768)) + (ax3_0_1_82 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_165 = 0; ax0_165 < 4; ++ax0_165) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_165], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_165 * 768)) + (ax3_0_1_82 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_82 = 0; ax1_0_2_82 < 4; ++ax1_0_2_82) {
      for (int ax2_0_2_82 = 0; ax2_0_2_82 < 4; ++ax2_0_2_82) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_82 * 4) + ax2_0_2_82)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_82], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_82], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_82 * 4) + ax2_0_2_82)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_166 = 0; ax0_ax1_fused_0_166 < 4; ++ax0_ax1_fused_0_166) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_166 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_166 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2656));
  }
  for (int ax0_ax1_fused_0_167 = 0; ax0_ax1_fused_0_167 < 4; ++ax0_ax1_fused_0_167) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_167 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_167 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2656));
  }
  __syncthreads();
  for (int ax3_0_1_83 = 0; ax3_0_1_83 < 2; ++ax3_0_1_83) {
    for (int ax0_166 = 0; ax0_166 < 4; ++ax0_166) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_166], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_166 * 768)) + (ax3_0_1_83 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_167 = 0; ax0_167 < 4; ++ax0_167) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_167], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_167 * 768)) + (ax3_0_1_83 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_83 = 0; ax1_0_2_83 < 4; ++ax1_0_2_83) {
      for (int ax2_0_2_83 = 0; ax2_0_2_83 < 4; ++ax2_0_2_83) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_83 * 4) + ax2_0_2_83)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_83], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_83], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_83 * 4) + ax2_0_2_83)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_168 = 0; ax0_ax1_fused_0_168 < 4; ++ax0_ax1_fused_0_168) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_168 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_168 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2688));
  }
  for (int ax0_ax1_fused_0_169 = 0; ax0_ax1_fused_0_169 < 4; ++ax0_ax1_fused_0_169) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_169 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_169 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2688));
  }
  __syncthreads();
  for (int ax3_0_1_84 = 0; ax3_0_1_84 < 2; ++ax3_0_1_84) {
    for (int ax0_168 = 0; ax0_168 < 4; ++ax0_168) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_168], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_168 * 768)) + (ax3_0_1_84 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_169 = 0; ax0_169 < 4; ++ax0_169) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_169], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_169 * 768)) + (ax3_0_1_84 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_84 = 0; ax1_0_2_84 < 4; ++ax1_0_2_84) {
      for (int ax2_0_2_84 = 0; ax2_0_2_84 < 4; ++ax2_0_2_84) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_84 * 4) + ax2_0_2_84)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_84], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_84], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_84 * 4) + ax2_0_2_84)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_170 = 0; ax0_ax1_fused_0_170 < 4; ++ax0_ax1_fused_0_170) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_170 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_170 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2720));
  }
  for (int ax0_ax1_fused_0_171 = 0; ax0_ax1_fused_0_171 < 4; ++ax0_ax1_fused_0_171) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_171 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_171 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2720));
  }
  __syncthreads();
  for (int ax3_0_1_85 = 0; ax3_0_1_85 < 2; ++ax3_0_1_85) {
    for (int ax0_170 = 0; ax0_170 < 4; ++ax0_170) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_170], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_170 * 768)) + (ax3_0_1_85 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_171 = 0; ax0_171 < 4; ++ax0_171) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_171], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_171 * 768)) + (ax3_0_1_85 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_85 = 0; ax1_0_2_85 < 4; ++ax1_0_2_85) {
      for (int ax2_0_2_85 = 0; ax2_0_2_85 < 4; ++ax2_0_2_85) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_85 * 4) + ax2_0_2_85)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_85], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_85], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_85 * 4) + ax2_0_2_85)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_172 = 0; ax0_ax1_fused_0_172 < 4; ++ax0_ax1_fused_0_172) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_172 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_172 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2752));
  }
  for (int ax0_ax1_fused_0_173 = 0; ax0_ax1_fused_0_173 < 4; ++ax0_ax1_fused_0_173) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_173 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_173 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2752));
  }
  __syncthreads();
  for (int ax3_0_1_86 = 0; ax3_0_1_86 < 2; ++ax3_0_1_86) {
    for (int ax0_172 = 0; ax0_172 < 4; ++ax0_172) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_172], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_172 * 768)) + (ax3_0_1_86 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_173 = 0; ax0_173 < 4; ++ax0_173) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_173], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_173 * 768)) + (ax3_0_1_86 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_86 = 0; ax1_0_2_86 < 4; ++ax1_0_2_86) {
      for (int ax2_0_2_86 = 0; ax2_0_2_86 < 4; ++ax2_0_2_86) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_86 * 4) + ax2_0_2_86)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_86], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_86], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_86 * 4) + ax2_0_2_86)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_174 = 0; ax0_ax1_fused_0_174 < 4; ++ax0_ax1_fused_0_174) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_174 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_174 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2784));
  }
  for (int ax0_ax1_fused_0_175 = 0; ax0_ax1_fused_0_175 < 4; ++ax0_ax1_fused_0_175) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_175 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_175 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2784));
  }
  __syncthreads();
  for (int ax3_0_1_87 = 0; ax3_0_1_87 < 2; ++ax3_0_1_87) {
    for (int ax0_174 = 0; ax0_174 < 4; ++ax0_174) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_174], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_174 * 768)) + (ax3_0_1_87 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_175 = 0; ax0_175 < 4; ++ax0_175) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_175], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_175 * 768)) + (ax3_0_1_87 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_87 = 0; ax1_0_2_87 < 4; ++ax1_0_2_87) {
      for (int ax2_0_2_87 = 0; ax2_0_2_87 < 4; ++ax2_0_2_87) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_87 * 4) + ax2_0_2_87)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_87], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_87], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_87 * 4) + ax2_0_2_87)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_176 = 0; ax0_ax1_fused_0_176 < 4; ++ax0_ax1_fused_0_176) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_176 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_176 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2816));
  }
  for (int ax0_ax1_fused_0_177 = 0; ax0_ax1_fused_0_177 < 4; ++ax0_ax1_fused_0_177) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_177 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_177 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2816));
  }
  __syncthreads();
  for (int ax3_0_1_88 = 0; ax3_0_1_88 < 2; ++ax3_0_1_88) {
    for (int ax0_176 = 0; ax0_176 < 4; ++ax0_176) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_176], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_176 * 768)) + (ax3_0_1_88 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_177 = 0; ax0_177 < 4; ++ax0_177) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_177], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_177 * 768)) + (ax3_0_1_88 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_88 = 0; ax1_0_2_88 < 4; ++ax1_0_2_88) {
      for (int ax2_0_2_88 = 0; ax2_0_2_88 < 4; ++ax2_0_2_88) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_88 * 4) + ax2_0_2_88)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_88], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_88], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_88 * 4) + ax2_0_2_88)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_178 = 0; ax0_ax1_fused_0_178 < 4; ++ax0_ax1_fused_0_178) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_178 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_178 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2848));
  }
  for (int ax0_ax1_fused_0_179 = 0; ax0_ax1_fused_0_179 < 4; ++ax0_ax1_fused_0_179) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_179 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_179 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2848));
  }
  __syncthreads();
  for (int ax3_0_1_89 = 0; ax3_0_1_89 < 2; ++ax3_0_1_89) {
    for (int ax0_178 = 0; ax0_178 < 4; ++ax0_178) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_178], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_178 * 768)) + (ax3_0_1_89 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_179 = 0; ax0_179 < 4; ++ax0_179) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_179], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_179 * 768)) + (ax3_0_1_89 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_89 = 0; ax1_0_2_89 < 4; ++ax1_0_2_89) {
      for (int ax2_0_2_89 = 0; ax2_0_2_89 < 4; ++ax2_0_2_89) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_89 * 4) + ax2_0_2_89)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_89], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_89], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_89 * 4) + ax2_0_2_89)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_180 = 0; ax0_ax1_fused_0_180 < 4; ++ax0_ax1_fused_0_180) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_180 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_180 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2880));
  }
  for (int ax0_ax1_fused_0_181 = 0; ax0_ax1_fused_0_181 < 4; ++ax0_ax1_fused_0_181) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_181 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_181 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2880));
  }
  __syncthreads();
  for (int ax3_0_1_90 = 0; ax3_0_1_90 < 2; ++ax3_0_1_90) {
    for (int ax0_180 = 0; ax0_180 < 4; ++ax0_180) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_180], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_180 * 768)) + (ax3_0_1_90 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_181 = 0; ax0_181 < 4; ++ax0_181) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_181], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_181 * 768)) + (ax3_0_1_90 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_90 = 0; ax1_0_2_90 < 4; ++ax1_0_2_90) {
      for (int ax2_0_2_90 = 0; ax2_0_2_90 < 4; ++ax2_0_2_90) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_90 * 4) + ax2_0_2_90)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_90], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_90], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_90 * 4) + ax2_0_2_90)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_182 = 0; ax0_ax1_fused_0_182 < 4; ++ax0_ax1_fused_0_182) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_182 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_182 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2912));
  }
  for (int ax0_ax1_fused_0_183 = 0; ax0_ax1_fused_0_183 < 4; ++ax0_ax1_fused_0_183) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_183 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_183 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2912));
  }
  __syncthreads();
  for (int ax3_0_1_91 = 0; ax3_0_1_91 < 2; ++ax3_0_1_91) {
    for (int ax0_182 = 0; ax0_182 < 4; ++ax0_182) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_182], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_182 * 768)) + (ax3_0_1_91 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_183 = 0; ax0_183 < 4; ++ax0_183) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_183], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_183 * 768)) + (ax3_0_1_91 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_91 = 0; ax1_0_2_91 < 4; ++ax1_0_2_91) {
      for (int ax2_0_2_91 = 0; ax2_0_2_91 < 4; ++ax2_0_2_91) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_91 * 4) + ax2_0_2_91)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_91], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_91], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_91 * 4) + ax2_0_2_91)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_184 = 0; ax0_ax1_fused_0_184 < 4; ++ax0_ax1_fused_0_184) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_184 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_184 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2944));
  }
  for (int ax0_ax1_fused_0_185 = 0; ax0_ax1_fused_0_185 < 4; ++ax0_ax1_fused_0_185) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_185 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_185 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2944));
  }
  __syncthreads();
  for (int ax3_0_1_92 = 0; ax3_0_1_92 < 2; ++ax3_0_1_92) {
    for (int ax0_184 = 0; ax0_184 < 4; ++ax0_184) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_184], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_184 * 768)) + (ax3_0_1_92 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_185 = 0; ax0_185 < 4; ++ax0_185) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_185], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_185 * 768)) + (ax3_0_1_92 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_92 = 0; ax1_0_2_92 < 4; ++ax1_0_2_92) {
      for (int ax2_0_2_92 = 0; ax2_0_2_92 < 4; ++ax2_0_2_92) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_92 * 4) + ax2_0_2_92)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_92], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_92], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_92 * 4) + ax2_0_2_92)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_186 = 0; ax0_ax1_fused_0_186 < 4; ++ax0_ax1_fused_0_186) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_186 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_186 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2976));
  }
  for (int ax0_ax1_fused_0_187 = 0; ax0_ax1_fused_0_187 < 4; ++ax0_ax1_fused_0_187) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_187 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_187 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2976));
  }
  __syncthreads();
  for (int ax3_0_1_93 = 0; ax3_0_1_93 < 2; ++ax3_0_1_93) {
    for (int ax0_186 = 0; ax0_186 < 4; ++ax0_186) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_186], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_186 * 768)) + (ax3_0_1_93 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_187 = 0; ax0_187 < 4; ++ax0_187) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_187], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_187 * 768)) + (ax3_0_1_93 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_93 = 0; ax1_0_2_93 < 4; ++ax1_0_2_93) {
      for (int ax2_0_2_93 = 0; ax2_0_2_93 < 4; ++ax2_0_2_93) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_93 * 4) + ax2_0_2_93)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_93], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_93], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_93 * 4) + ax2_0_2_93)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_188 = 0; ax0_ax1_fused_0_188 < 4; ++ax0_ax1_fused_0_188) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_188 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_188 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3008));
  }
  for (int ax0_ax1_fused_0_189 = 0; ax0_ax1_fused_0_189 < 4; ++ax0_ax1_fused_0_189) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_189 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_189 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3008));
  }
  __syncthreads();
  for (int ax3_0_1_94 = 0; ax3_0_1_94 < 2; ++ax3_0_1_94) {
    for (int ax0_188 = 0; ax0_188 < 4; ++ax0_188) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_188], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_188 * 768)) + (ax3_0_1_94 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_189 = 0; ax0_189 < 4; ++ax0_189) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_189], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_189 * 768)) + (ax3_0_1_94 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_94 = 0; ax1_0_2_94 < 4; ++ax1_0_2_94) {
      for (int ax2_0_2_94 = 0; ax2_0_2_94 < 4; ++ax2_0_2_94) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_94 * 4) + ax2_0_2_94)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_94], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_94], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_94 * 4) + ax2_0_2_94)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_190 = 0; ax0_ax1_fused_0_190 < 4; ++ax0_ax1_fused_0_190) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_190 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_190 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3040));
  }
  for (int ax0_ax1_fused_0_191 = 0; ax0_ax1_fused_0_191 < 4; ++ax0_ax1_fused_0_191) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_191 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_191 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3040));
  }
  __syncthreads();
  for (int ax3_0_1_95 = 0; ax3_0_1_95 < 2; ++ax3_0_1_95) {
    for (int ax0_190 = 0; ax0_190 < 4; ++ax0_190) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_190], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_190 * 768)) + (ax3_0_1_95 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_191 = 0; ax0_191 < 4; ++ax0_191) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_191], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_191 * 768)) + (ax3_0_1_95 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_95 = 0; ax1_0_2_95 < 4; ++ax1_0_2_95) {
      for (int ax2_0_2_95 = 0; ax2_0_2_95 < 4; ++ax2_0_2_95) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_95 * 4) + ax2_0_2_95)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_95], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_95], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_95 * 4) + ax2_0_2_95)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_192 = 0; ax0_ax1_fused_0_192 < 4; ++ax0_ax1_fused_0_192) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_192 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_192 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3072));
  }
  for (int ax0_ax1_fused_0_193 = 0; ax0_ax1_fused_0_193 < 4; ++ax0_ax1_fused_0_193) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_193 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_193 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3072));
  }
  __syncthreads();
  for (int ax3_0_1_96 = 0; ax3_0_1_96 < 2; ++ax3_0_1_96) {
    for (int ax0_192 = 0; ax0_192 < 4; ++ax0_192) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_192], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_192 * 768)) + (ax3_0_1_96 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_193 = 0; ax0_193 < 4; ++ax0_193) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_193], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_193 * 768)) + (ax3_0_1_96 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_96 = 0; ax1_0_2_96 < 4; ++ax1_0_2_96) {
      for (int ax2_0_2_96 = 0; ax2_0_2_96 < 4; ++ax2_0_2_96) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_96 * 4) + ax2_0_2_96)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_96], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_96], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_96 * 4) + ax2_0_2_96)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_194 = 0; ax0_ax1_fused_0_194 < 4; ++ax0_ax1_fused_0_194) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_194 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_194 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3104));
  }
  for (int ax0_ax1_fused_0_195 = 0; ax0_ax1_fused_0_195 < 4; ++ax0_ax1_fused_0_195) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_195 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_195 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3104));
  }
  __syncthreads();
  for (int ax3_0_1_97 = 0; ax3_0_1_97 < 2; ++ax3_0_1_97) {
    for (int ax0_194 = 0; ax0_194 < 4; ++ax0_194) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_194], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_194 * 768)) + (ax3_0_1_97 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_195 = 0; ax0_195 < 4; ++ax0_195) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_195], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_195 * 768)) + (ax3_0_1_97 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_97 = 0; ax1_0_2_97 < 4; ++ax1_0_2_97) {
      for (int ax2_0_2_97 = 0; ax2_0_2_97 < 4; ++ax2_0_2_97) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_97 * 4) + ax2_0_2_97)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_97], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_97], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_97 * 4) + ax2_0_2_97)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_196 = 0; ax0_ax1_fused_0_196 < 4; ++ax0_ax1_fused_0_196) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_196 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_196 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3136));
  }
  for (int ax0_ax1_fused_0_197 = 0; ax0_ax1_fused_0_197 < 4; ++ax0_ax1_fused_0_197) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_197 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_197 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3136));
  }
  __syncthreads();
  for (int ax3_0_1_98 = 0; ax3_0_1_98 < 2; ++ax3_0_1_98) {
    for (int ax0_196 = 0; ax0_196 < 4; ++ax0_196) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_196], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_196 * 768)) + (ax3_0_1_98 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_197 = 0; ax0_197 < 4; ++ax0_197) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_197], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_197 * 768)) + (ax3_0_1_98 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_98 = 0; ax1_0_2_98 < 4; ++ax1_0_2_98) {
      for (int ax2_0_2_98 = 0; ax2_0_2_98 < 4; ++ax2_0_2_98) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_98 * 4) + ax2_0_2_98)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_98], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_98], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_98 * 4) + ax2_0_2_98)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_198 = 0; ax0_ax1_fused_0_198 < 4; ++ax0_ax1_fused_0_198) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_198 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_198 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3168));
  }
  for (int ax0_ax1_fused_0_199 = 0; ax0_ax1_fused_0_199 < 4; ++ax0_ax1_fused_0_199) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_199 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_199 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3168));
  }
  __syncthreads();
  for (int ax3_0_1_99 = 0; ax3_0_1_99 < 2; ++ax3_0_1_99) {
    for (int ax0_198 = 0; ax0_198 < 4; ++ax0_198) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_198], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_198 * 768)) + (ax3_0_1_99 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_199 = 0; ax0_199 < 4; ++ax0_199) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_199], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_199 * 768)) + (ax3_0_1_99 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_99 = 0; ax1_0_2_99 < 4; ++ax1_0_2_99) {
      for (int ax2_0_2_99 = 0; ax2_0_2_99 < 4; ++ax2_0_2_99) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_99 * 4) + ax2_0_2_99)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_99], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_99], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_99 * 4) + ax2_0_2_99)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_200 = 0; ax0_ax1_fused_0_200 < 4; ++ax0_ax1_fused_0_200) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_200 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_200 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3200));
  }
  for (int ax0_ax1_fused_0_201 = 0; ax0_ax1_fused_0_201 < 4; ++ax0_ax1_fused_0_201) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_201 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_201 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3200));
  }
  __syncthreads();
  for (int ax3_0_1_100 = 0; ax3_0_1_100 < 2; ++ax3_0_1_100) {
    for (int ax0_200 = 0; ax0_200 < 4; ++ax0_200) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_200], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_200 * 768)) + (ax3_0_1_100 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_201 = 0; ax0_201 < 4; ++ax0_201) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_201], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_201 * 768)) + (ax3_0_1_100 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_100 = 0; ax1_0_2_100 < 4; ++ax1_0_2_100) {
      for (int ax2_0_2_100 = 0; ax2_0_2_100 < 4; ++ax2_0_2_100) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_100 * 4) + ax2_0_2_100)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_100], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_100], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_100 * 4) + ax2_0_2_100)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_202 = 0; ax0_ax1_fused_0_202 < 4; ++ax0_ax1_fused_0_202) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_202 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_202 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3232));
  }
  for (int ax0_ax1_fused_0_203 = 0; ax0_ax1_fused_0_203 < 4; ++ax0_ax1_fused_0_203) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_203 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_203 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3232));
  }
  __syncthreads();
  for (int ax3_0_1_101 = 0; ax3_0_1_101 < 2; ++ax3_0_1_101) {
    for (int ax0_202 = 0; ax0_202 < 4; ++ax0_202) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_202], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_202 * 768)) + (ax3_0_1_101 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_203 = 0; ax0_203 < 4; ++ax0_203) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_203], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_203 * 768)) + (ax3_0_1_101 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_101 = 0; ax1_0_2_101 < 4; ++ax1_0_2_101) {
      for (int ax2_0_2_101 = 0; ax2_0_2_101 < 4; ++ax2_0_2_101) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_101 * 4) + ax2_0_2_101)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_101], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_101], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_101 * 4) + ax2_0_2_101)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_204 = 0; ax0_ax1_fused_0_204 < 4; ++ax0_ax1_fused_0_204) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_204 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_204 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3264));
  }
  for (int ax0_ax1_fused_0_205 = 0; ax0_ax1_fused_0_205 < 4; ++ax0_ax1_fused_0_205) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_205 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_205 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3264));
  }
  __syncthreads();
  for (int ax3_0_1_102 = 0; ax3_0_1_102 < 2; ++ax3_0_1_102) {
    for (int ax0_204 = 0; ax0_204 < 4; ++ax0_204) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_204], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_204 * 768)) + (ax3_0_1_102 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_205 = 0; ax0_205 < 4; ++ax0_205) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_205], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_205 * 768)) + (ax3_0_1_102 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_102 = 0; ax1_0_2_102 < 4; ++ax1_0_2_102) {
      for (int ax2_0_2_102 = 0; ax2_0_2_102 < 4; ++ax2_0_2_102) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_102 * 4) + ax2_0_2_102)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_102], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_102], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_102 * 4) + ax2_0_2_102)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_206 = 0; ax0_ax1_fused_0_206 < 4; ++ax0_ax1_fused_0_206) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_206 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_206 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3296));
  }
  for (int ax0_ax1_fused_0_207 = 0; ax0_ax1_fused_0_207 < 4; ++ax0_ax1_fused_0_207) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_207 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_207 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3296));
  }
  __syncthreads();
  for (int ax3_0_1_103 = 0; ax3_0_1_103 < 2; ++ax3_0_1_103) {
    for (int ax0_206 = 0; ax0_206 < 4; ++ax0_206) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_206], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_206 * 768)) + (ax3_0_1_103 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_207 = 0; ax0_207 < 4; ++ax0_207) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_207], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_207 * 768)) + (ax3_0_1_103 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_103 = 0; ax1_0_2_103 < 4; ++ax1_0_2_103) {
      for (int ax2_0_2_103 = 0; ax2_0_2_103 < 4; ++ax2_0_2_103) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_103 * 4) + ax2_0_2_103)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_103], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_103], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_103 * 4) + ax2_0_2_103)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_208 = 0; ax0_ax1_fused_0_208 < 4; ++ax0_ax1_fused_0_208) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_208 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_208 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3328));
  }
  for (int ax0_ax1_fused_0_209 = 0; ax0_ax1_fused_0_209 < 4; ++ax0_ax1_fused_0_209) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_209 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_209 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3328));
  }
  __syncthreads();
  for (int ax3_0_1_104 = 0; ax3_0_1_104 < 2; ++ax3_0_1_104) {
    for (int ax0_208 = 0; ax0_208 < 4; ++ax0_208) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_208], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_208 * 768)) + (ax3_0_1_104 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_209 = 0; ax0_209 < 4; ++ax0_209) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_209], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_209 * 768)) + (ax3_0_1_104 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_104 = 0; ax1_0_2_104 < 4; ++ax1_0_2_104) {
      for (int ax2_0_2_104 = 0; ax2_0_2_104 < 4; ++ax2_0_2_104) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_104 * 4) + ax2_0_2_104)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_104], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_104], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_104 * 4) + ax2_0_2_104)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_210 = 0; ax0_ax1_fused_0_210 < 4; ++ax0_ax1_fused_0_210) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_210 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_210 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3360));
  }
  for (int ax0_ax1_fused_0_211 = 0; ax0_ax1_fused_0_211 < 4; ++ax0_ax1_fused_0_211) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_211 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_211 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3360));
  }
  __syncthreads();
  for (int ax3_0_1_105 = 0; ax3_0_1_105 < 2; ++ax3_0_1_105) {
    for (int ax0_210 = 0; ax0_210 < 4; ++ax0_210) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_210], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_210 * 768)) + (ax3_0_1_105 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_211 = 0; ax0_211 < 4; ++ax0_211) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_211], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_211 * 768)) + (ax3_0_1_105 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_105 = 0; ax1_0_2_105 < 4; ++ax1_0_2_105) {
      for (int ax2_0_2_105 = 0; ax2_0_2_105 < 4; ++ax2_0_2_105) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_105 * 4) + ax2_0_2_105)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_105], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_105], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_105 * 4) + ax2_0_2_105)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_212 = 0; ax0_ax1_fused_0_212 < 4; ++ax0_ax1_fused_0_212) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_212 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_212 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3392));
  }
  for (int ax0_ax1_fused_0_213 = 0; ax0_ax1_fused_0_213 < 4; ++ax0_ax1_fused_0_213) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_213 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_213 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3392));
  }
  __syncthreads();
  for (int ax3_0_1_106 = 0; ax3_0_1_106 < 2; ++ax3_0_1_106) {
    for (int ax0_212 = 0; ax0_212 < 4; ++ax0_212) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_212], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_212 * 768)) + (ax3_0_1_106 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_213 = 0; ax0_213 < 4; ++ax0_213) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_213], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_213 * 768)) + (ax3_0_1_106 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_106 = 0; ax1_0_2_106 < 4; ++ax1_0_2_106) {
      for (int ax2_0_2_106 = 0; ax2_0_2_106 < 4; ++ax2_0_2_106) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_106 * 4) + ax2_0_2_106)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_106], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_106], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_106 * 4) + ax2_0_2_106)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_214 = 0; ax0_ax1_fused_0_214 < 4; ++ax0_ax1_fused_0_214) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_214 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_214 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3424));
  }
  for (int ax0_ax1_fused_0_215 = 0; ax0_ax1_fused_0_215 < 4; ++ax0_ax1_fused_0_215) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_215 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_215 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3424));
  }
  __syncthreads();
  for (int ax3_0_1_107 = 0; ax3_0_1_107 < 2; ++ax3_0_1_107) {
    for (int ax0_214 = 0; ax0_214 < 4; ++ax0_214) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_214], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_214 * 768)) + (ax3_0_1_107 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_215 = 0; ax0_215 < 4; ++ax0_215) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_215], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_215 * 768)) + (ax3_0_1_107 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_107 = 0; ax1_0_2_107 < 4; ++ax1_0_2_107) {
      for (int ax2_0_2_107 = 0; ax2_0_2_107 < 4; ++ax2_0_2_107) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_107 * 4) + ax2_0_2_107)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_107], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_107], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_107 * 4) + ax2_0_2_107)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_216 = 0; ax0_ax1_fused_0_216 < 4; ++ax0_ax1_fused_0_216) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_216 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_216 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3456));
  }
  for (int ax0_ax1_fused_0_217 = 0; ax0_ax1_fused_0_217 < 4; ++ax0_ax1_fused_0_217) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_217 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_217 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3456));
  }
  __syncthreads();
  for (int ax3_0_1_108 = 0; ax3_0_1_108 < 2; ++ax3_0_1_108) {
    for (int ax0_216 = 0; ax0_216 < 4; ++ax0_216) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_216], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_216 * 768)) + (ax3_0_1_108 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_217 = 0; ax0_217 < 4; ++ax0_217) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_217], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_217 * 768)) + (ax3_0_1_108 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_108 = 0; ax1_0_2_108 < 4; ++ax1_0_2_108) {
      for (int ax2_0_2_108 = 0; ax2_0_2_108 < 4; ++ax2_0_2_108) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_108 * 4) + ax2_0_2_108)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_108], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_108], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_108 * 4) + ax2_0_2_108)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_218 = 0; ax0_ax1_fused_0_218 < 4; ++ax0_ax1_fused_0_218) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_218 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_218 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3488));
  }
  for (int ax0_ax1_fused_0_219 = 0; ax0_ax1_fused_0_219 < 4; ++ax0_ax1_fused_0_219) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_219 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_219 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3488));
  }
  __syncthreads();
  for (int ax3_0_1_109 = 0; ax3_0_1_109 < 2; ++ax3_0_1_109) {
    for (int ax0_218 = 0; ax0_218 < 4; ++ax0_218) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_218], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_218 * 768)) + (ax3_0_1_109 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_219 = 0; ax0_219 < 4; ++ax0_219) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_219], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_219 * 768)) + (ax3_0_1_109 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_109 = 0; ax1_0_2_109 < 4; ++ax1_0_2_109) {
      for (int ax2_0_2_109 = 0; ax2_0_2_109 < 4; ++ax2_0_2_109) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_109 * 4) + ax2_0_2_109)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_109], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_109], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_109 * 4) + ax2_0_2_109)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_220 = 0; ax0_ax1_fused_0_220 < 4; ++ax0_ax1_fused_0_220) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_220 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_220 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3520));
  }
  for (int ax0_ax1_fused_0_221 = 0; ax0_ax1_fused_0_221 < 4; ++ax0_ax1_fused_0_221) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_221 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_221 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3520));
  }
  __syncthreads();
  for (int ax3_0_1_110 = 0; ax3_0_1_110 < 2; ++ax3_0_1_110) {
    for (int ax0_220 = 0; ax0_220 < 4; ++ax0_220) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_220], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_220 * 768)) + (ax3_0_1_110 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_221 = 0; ax0_221 < 4; ++ax0_221) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_221], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_221 * 768)) + (ax3_0_1_110 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_110 = 0; ax1_0_2_110 < 4; ++ax1_0_2_110) {
      for (int ax2_0_2_110 = 0; ax2_0_2_110 < 4; ++ax2_0_2_110) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_110 * 4) + ax2_0_2_110)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_110], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_110], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_110 * 4) + ax2_0_2_110)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_222 = 0; ax0_ax1_fused_0_222 < 4; ++ax0_ax1_fused_0_222) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_222 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_222 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3552));
  }
  for (int ax0_ax1_fused_0_223 = 0; ax0_ax1_fused_0_223 < 4; ++ax0_ax1_fused_0_223) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_223 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_223 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3552));
  }
  __syncthreads();
  for (int ax3_0_1_111 = 0; ax3_0_1_111 < 2; ++ax3_0_1_111) {
    for (int ax0_222 = 0; ax0_222 < 4; ++ax0_222) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_222], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_222 * 768)) + (ax3_0_1_111 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_223 = 0; ax0_223 < 4; ++ax0_223) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_223], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_223 * 768)) + (ax3_0_1_111 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_111 = 0; ax1_0_2_111 < 4; ++ax1_0_2_111) {
      for (int ax2_0_2_111 = 0; ax2_0_2_111 < 4; ++ax2_0_2_111) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_111 * 4) + ax2_0_2_111)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_111], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_111], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_111 * 4) + ax2_0_2_111)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_224 = 0; ax0_ax1_fused_0_224 < 4; ++ax0_ax1_fused_0_224) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_224 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_224 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3584));
  }
  for (int ax0_ax1_fused_0_225 = 0; ax0_ax1_fused_0_225 < 4; ++ax0_ax1_fused_0_225) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_225 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_225 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3584));
  }
  __syncthreads();
  for (int ax3_0_1_112 = 0; ax3_0_1_112 < 2; ++ax3_0_1_112) {
    for (int ax0_224 = 0; ax0_224 < 4; ++ax0_224) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_224], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_224 * 768)) + (ax3_0_1_112 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_225 = 0; ax0_225 < 4; ++ax0_225) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_225], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_225 * 768)) + (ax3_0_1_112 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_112 = 0; ax1_0_2_112 < 4; ++ax1_0_2_112) {
      for (int ax2_0_2_112 = 0; ax2_0_2_112 < 4; ++ax2_0_2_112) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_112 * 4) + ax2_0_2_112)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_112], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_112], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_112 * 4) + ax2_0_2_112)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_226 = 0; ax0_ax1_fused_0_226 < 4; ++ax0_ax1_fused_0_226) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_226 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_226 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3616));
  }
  for (int ax0_ax1_fused_0_227 = 0; ax0_ax1_fused_0_227 < 4; ++ax0_ax1_fused_0_227) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_227 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_227 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3616));
  }
  __syncthreads();
  for (int ax3_0_1_113 = 0; ax3_0_1_113 < 2; ++ax3_0_1_113) {
    for (int ax0_226 = 0; ax0_226 < 4; ++ax0_226) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_226], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_226 * 768)) + (ax3_0_1_113 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_227 = 0; ax0_227 < 4; ++ax0_227) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_227], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_227 * 768)) + (ax3_0_1_113 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_113 = 0; ax1_0_2_113 < 4; ++ax1_0_2_113) {
      for (int ax2_0_2_113 = 0; ax2_0_2_113 < 4; ++ax2_0_2_113) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_113 * 4) + ax2_0_2_113)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_113], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_113], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_113 * 4) + ax2_0_2_113)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_228 = 0; ax0_ax1_fused_0_228 < 4; ++ax0_ax1_fused_0_228) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_228 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_228 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3648));
  }
  for (int ax0_ax1_fused_0_229 = 0; ax0_ax1_fused_0_229 < 4; ++ax0_ax1_fused_0_229) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_229 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_229 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3648));
  }
  __syncthreads();
  for (int ax3_0_1_114 = 0; ax3_0_1_114 < 2; ++ax3_0_1_114) {
    for (int ax0_228 = 0; ax0_228 < 4; ++ax0_228) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_228], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_228 * 768)) + (ax3_0_1_114 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_229 = 0; ax0_229 < 4; ++ax0_229) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_229], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_229 * 768)) + (ax3_0_1_114 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_114 = 0; ax1_0_2_114 < 4; ++ax1_0_2_114) {
      for (int ax2_0_2_114 = 0; ax2_0_2_114 < 4; ++ax2_0_2_114) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_114 * 4) + ax2_0_2_114)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_114], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_114], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_114 * 4) + ax2_0_2_114)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_230 = 0; ax0_ax1_fused_0_230 < 4; ++ax0_ax1_fused_0_230) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_230 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_230 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3680));
  }
  for (int ax0_ax1_fused_0_231 = 0; ax0_ax1_fused_0_231 < 4; ++ax0_ax1_fused_0_231) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_231 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_231 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3680));
  }
  __syncthreads();
  for (int ax3_0_1_115 = 0; ax3_0_1_115 < 2; ++ax3_0_1_115) {
    for (int ax0_230 = 0; ax0_230 < 4; ++ax0_230) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_230], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_230 * 768)) + (ax3_0_1_115 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_231 = 0; ax0_231 < 4; ++ax0_231) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_231], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_231 * 768)) + (ax3_0_1_115 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_115 = 0; ax1_0_2_115 < 4; ++ax1_0_2_115) {
      for (int ax2_0_2_115 = 0; ax2_0_2_115 < 4; ++ax2_0_2_115) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_115 * 4) + ax2_0_2_115)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_115], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_115], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_115 * 4) + ax2_0_2_115)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_232 = 0; ax0_ax1_fused_0_232 < 4; ++ax0_ax1_fused_0_232) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_232 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_232 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3712));
  }
  for (int ax0_ax1_fused_0_233 = 0; ax0_ax1_fused_0_233 < 4; ++ax0_ax1_fused_0_233) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_233 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_233 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3712));
  }
  __syncthreads();
  for (int ax3_0_1_116 = 0; ax3_0_1_116 < 2; ++ax3_0_1_116) {
    for (int ax0_232 = 0; ax0_232 < 4; ++ax0_232) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_232], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_232 * 768)) + (ax3_0_1_116 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_233 = 0; ax0_233 < 4; ++ax0_233) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_233], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_233 * 768)) + (ax3_0_1_116 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_116 = 0; ax1_0_2_116 < 4; ++ax1_0_2_116) {
      for (int ax2_0_2_116 = 0; ax2_0_2_116 < 4; ++ax2_0_2_116) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_116 * 4) + ax2_0_2_116)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_116], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_116], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_116 * 4) + ax2_0_2_116)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_234 = 0; ax0_ax1_fused_0_234 < 4; ++ax0_ax1_fused_0_234) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_234 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_234 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3744));
  }
  for (int ax0_ax1_fused_0_235 = 0; ax0_ax1_fused_0_235 < 4; ++ax0_ax1_fused_0_235) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_235 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_235 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3744));
  }
  __syncthreads();
  for (int ax3_0_1_117 = 0; ax3_0_1_117 < 2; ++ax3_0_1_117) {
    for (int ax0_234 = 0; ax0_234 < 4; ++ax0_234) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_234], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_234 * 768)) + (ax3_0_1_117 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_235 = 0; ax0_235 < 4; ++ax0_235) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_235], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_235 * 768)) + (ax3_0_1_117 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_117 = 0; ax1_0_2_117 < 4; ++ax1_0_2_117) {
      for (int ax2_0_2_117 = 0; ax2_0_2_117 < 4; ++ax2_0_2_117) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_117 * 4) + ax2_0_2_117)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_117], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_117], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_117 * 4) + ax2_0_2_117)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_236 = 0; ax0_ax1_fused_0_236 < 4; ++ax0_ax1_fused_0_236) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_236 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_236 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3776));
  }
  for (int ax0_ax1_fused_0_237 = 0; ax0_ax1_fused_0_237 < 4; ++ax0_ax1_fused_0_237) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_237 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_237 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3776));
  }
  __syncthreads();
  for (int ax3_0_1_118 = 0; ax3_0_1_118 < 2; ++ax3_0_1_118) {
    for (int ax0_236 = 0; ax0_236 < 4; ++ax0_236) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_236], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_236 * 768)) + (ax3_0_1_118 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_237 = 0; ax0_237 < 4; ++ax0_237) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_237], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_237 * 768)) + (ax3_0_1_118 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_118 = 0; ax1_0_2_118 < 4; ++ax1_0_2_118) {
      for (int ax2_0_2_118 = 0; ax2_0_2_118 < 4; ++ax2_0_2_118) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_118 * 4) + ax2_0_2_118)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_118], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_118], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_118 * 4) + ax2_0_2_118)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_238 = 0; ax0_ax1_fused_0_238 < 4; ++ax0_ax1_fused_0_238) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_238 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_238 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3808));
  }
  for (int ax0_ax1_fused_0_239 = 0; ax0_ax1_fused_0_239 < 4; ++ax0_ax1_fused_0_239) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_239 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_239 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3808));
  }
  __syncthreads();
  for (int ax3_0_1_119 = 0; ax3_0_1_119 < 2; ++ax3_0_1_119) {
    for (int ax0_238 = 0; ax0_238 < 4; ++ax0_238) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_238], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_238 * 768)) + (ax3_0_1_119 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_239 = 0; ax0_239 < 4; ++ax0_239) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_239], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_239 * 768)) + (ax3_0_1_119 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_119 = 0; ax1_0_2_119 < 4; ++ax1_0_2_119) {
      for (int ax2_0_2_119 = 0; ax2_0_2_119 < 4; ++ax2_0_2_119) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_119 * 4) + ax2_0_2_119)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_119], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_119], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_119 * 4) + ax2_0_2_119)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_240 = 0; ax0_ax1_fused_0_240 < 4; ++ax0_ax1_fused_0_240) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_240 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_240 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3840));
  }
  for (int ax0_ax1_fused_0_241 = 0; ax0_ax1_fused_0_241 < 4; ++ax0_ax1_fused_0_241) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_241 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_241 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3840));
  }
  __syncthreads();
  for (int ax3_0_1_120 = 0; ax3_0_1_120 < 2; ++ax3_0_1_120) {
    for (int ax0_240 = 0; ax0_240 < 4; ++ax0_240) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_240], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_240 * 768)) + (ax3_0_1_120 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_241 = 0; ax0_241 < 4; ++ax0_241) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_241], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_241 * 768)) + (ax3_0_1_120 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_120 = 0; ax1_0_2_120 < 4; ++ax1_0_2_120) {
      for (int ax2_0_2_120 = 0; ax2_0_2_120 < 4; ++ax2_0_2_120) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_120 * 4) + ax2_0_2_120)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_120], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_120], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_120 * 4) + ax2_0_2_120)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_242 = 0; ax0_ax1_fused_0_242 < 4; ++ax0_ax1_fused_0_242) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_242 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_242 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3872));
  }
  for (int ax0_ax1_fused_0_243 = 0; ax0_ax1_fused_0_243 < 4; ++ax0_ax1_fused_0_243) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_243 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_243 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3872));
  }
  __syncthreads();
  for (int ax3_0_1_121 = 0; ax3_0_1_121 < 2; ++ax3_0_1_121) {
    for (int ax0_242 = 0; ax0_242 < 4; ++ax0_242) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_242], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_242 * 768)) + (ax3_0_1_121 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_243 = 0; ax0_243 < 4; ++ax0_243) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_243], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_243 * 768)) + (ax3_0_1_121 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_121 = 0; ax1_0_2_121 < 4; ++ax1_0_2_121) {
      for (int ax2_0_2_121 = 0; ax2_0_2_121 < 4; ++ax2_0_2_121) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_121 * 4) + ax2_0_2_121)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_121], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_121], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_121 * 4) + ax2_0_2_121)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_244 = 0; ax0_ax1_fused_0_244 < 4; ++ax0_ax1_fused_0_244) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_244 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_244 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3904));
  }
  for (int ax0_ax1_fused_0_245 = 0; ax0_ax1_fused_0_245 < 4; ++ax0_ax1_fused_0_245) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_245 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_245 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3904));
  }
  __syncthreads();
  for (int ax3_0_1_122 = 0; ax3_0_1_122 < 2; ++ax3_0_1_122) {
    for (int ax0_244 = 0; ax0_244 < 4; ++ax0_244) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_244], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_244 * 768)) + (ax3_0_1_122 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_245 = 0; ax0_245 < 4; ++ax0_245) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_245], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_245 * 768)) + (ax3_0_1_122 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_122 = 0; ax1_0_2_122 < 4; ++ax1_0_2_122) {
      for (int ax2_0_2_122 = 0; ax2_0_2_122 < 4; ++ax2_0_2_122) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_122 * 4) + ax2_0_2_122)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_122], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_122], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_122 * 4) + ax2_0_2_122)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_246 = 0; ax0_ax1_fused_0_246 < 4; ++ax0_ax1_fused_0_246) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_246 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_246 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3936));
  }
  for (int ax0_ax1_fused_0_247 = 0; ax0_ax1_fused_0_247 < 4; ++ax0_ax1_fused_0_247) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_247 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_247 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3936));
  }
  __syncthreads();
  for (int ax3_0_1_123 = 0; ax3_0_1_123 < 2; ++ax3_0_1_123) {
    for (int ax0_246 = 0; ax0_246 < 4; ++ax0_246) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_246], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_246 * 768)) + (ax3_0_1_123 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_247 = 0; ax0_247 < 4; ++ax0_247) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_247], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_247 * 768)) + (ax3_0_1_123 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_123 = 0; ax1_0_2_123 < 4; ++ax1_0_2_123) {
      for (int ax2_0_2_123 = 0; ax2_0_2_123 < 4; ++ax2_0_2_123) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_123 * 4) + ax2_0_2_123)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_123], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_123], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_123 * 4) + ax2_0_2_123)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_248 = 0; ax0_ax1_fused_0_248 < 4; ++ax0_ax1_fused_0_248) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_248 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_248 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3968));
  }
  for (int ax0_ax1_fused_0_249 = 0; ax0_ax1_fused_0_249 < 4; ++ax0_ax1_fused_0_249) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_249 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_249 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3968));
  }
  __syncthreads();
  for (int ax3_0_1_124 = 0; ax3_0_1_124 < 2; ++ax3_0_1_124) {
    for (int ax0_248 = 0; ax0_248 < 4; ++ax0_248) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_248], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_248 * 768)) + (ax3_0_1_124 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_249 = 0; ax0_249 < 4; ++ax0_249) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_249], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_249 * 768)) + (ax3_0_1_124 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_124 = 0; ax1_0_2_124 < 4; ++ax1_0_2_124) {
      for (int ax2_0_2_124 = 0; ax2_0_2_124 < 4; ++ax2_0_2_124) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_124 * 4) + ax2_0_2_124)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_124], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_124], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_124 * 4) + ax2_0_2_124)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_250 = 0; ax0_ax1_fused_0_250 < 4; ++ax0_ax1_fused_0_250) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_250 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_250 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 4000));
  }
  for (int ax0_ax1_fused_0_251 = 0; ax0_ax1_fused_0_251 < 4; ++ax0_ax1_fused_0_251) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_251 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_251 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 4000));
  }
  __syncthreads();
  for (int ax3_0_1_125 = 0; ax3_0_1_125 < 2; ++ax3_0_1_125) {
    for (int ax0_250 = 0; ax0_250 < 4; ++ax0_250) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_250], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_250 * 768)) + (ax3_0_1_125 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_251 = 0; ax0_251 < 4; ++ax0_251) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_251], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_251 * 768)) + (ax3_0_1_125 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_125 = 0; ax1_0_2_125 < 4; ++ax1_0_2_125) {
      for (int ax2_0_2_125 = 0; ax2_0_2_125 < 4; ++ax2_0_2_125) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_125 * 4) + ax2_0_2_125)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_125], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_125], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_125 * 4) + ax2_0_2_125)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_252 = 0; ax0_ax1_fused_0_252 < 4; ++ax0_ax1_fused_0_252) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_252 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_252 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 4032));
  }
  for (int ax0_ax1_fused_0_253 = 0; ax0_ax1_fused_0_253 < 4; ++ax0_ax1_fused_0_253) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_253 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_253 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 4032));
  }
  __syncthreads();
  for (int ax3_0_1_126 = 0; ax3_0_1_126 < 2; ++ax3_0_1_126) {
    for (int ax0_252 = 0; ax0_252 < 4; ++ax0_252) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_252], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_252 * 768)) + (ax3_0_1_126 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_253 = 0; ax0_253 < 4; ++ax0_253) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_253], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_253 * 768)) + (ax3_0_1_126 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_126 = 0; ax1_0_2_126 < 4; ++ax1_0_2_126) {
      for (int ax2_0_2_126 = 0; ax2_0_2_126 < 4; ++ax2_0_2_126) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_126 * 4) + ax2_0_2_126)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_126], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_126], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_126 * 4) + ax2_0_2_126)]);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_254 = 0; ax0_ax1_fused_0_254 < 4; ++ax0_ax1_fused_0_254) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0_254 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + ((((((ax0_ax1_fused_0_254 * 131072) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 4064));
  }
  for (int ax0_ax1_fused_0_255 = 0; ax0_ax1_fused_0_255 < 4; ++ax0_ax1_fused_0_255) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_255 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + (((((((((int)blockIdx.x) * 524288) + (ax0_ax1_fused_0_255 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 4064));
  }
  __syncthreads();
  for (int ax3_0_1_127 = 0; ax3_0_1_127 < 2; ++ax3_0_1_127) {
    for (int ax0_254 = 0; ax0_254 < 4; ++ax0_254) {
      nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0_254], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0_254 * 768)) + (ax3_0_1_127 * 384)) + 6144)])), (int64_t)24);
    }
    for (int ax0_255 = 0; ax0_255 < 4; ++ax0_255) {
      nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_255], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_255 * 768)) + (ax3_0_1_127 * 384))])), (int64_t)24);
    }
    for (int ax1_0_2_127 = 0; ax1_0_2_127 < 4; ++ax1_0_2_127) {
      for (int ax2_0_2_127 = 0; ax2_0_2_127 < 4; ++ax2_0_2_127) {
        nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_127 * 4) + ax2_0_2_127)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_2_127], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_2_127], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_2_127 * 4) + ax2_0_2_127)]);
      }
    }
  }
  __syncthreads();
  for (int ax2_1 = 0; ax2_1 < 4; ++ax2_1) {
    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
      nvcuda::wmma::store_matrix_sync((&(((float*)buf_dyn_shmem)[((((((int)threadIdx.z) * 8192) + (ax2_1 * 2048)) + (((int)threadIdx.y) * 1024)) + (ax3_1 * 256))])), matmul_reindex_shared_dyn_wmma_accumulator[((ax2_1 * 4) + ax3_1)], (int64_t)16, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_256 = 0; ax0_ax1_fused_0_256 < 16; ++ax0_ax1_fused_0_256) {
    uint4 __1;
    ulonglong4 v_ = *(ulonglong4*)(((float*)buf_dyn_shmem) + (((((((((ax0_ax1_fused_0_256 * 8) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) >> 4) * 2048) + (((((int)threadIdx.x) & 15) >> 1) * 256)) + ((((((ax0_ax1_fused_0_256 * 8) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) & 15) * 16)) + ((((int)threadIdx.x) & 1) * 8)));
    ((half2*)(&(__1.x)))->x = (half)(((float2*)(&(v_.x)))->x);
    ((half2*)(&(__1.x)))->y = (half)(((float2*)(&(v_.x)))->y);
    ((half2*)(&(__1.y)))->x = (half)(((float2*)(&(v_.y)))->x);
    ((half2*)(&(__1.y)))->y = (half)(((float2*)(&(v_.y)))->y);
    ((half2*)(&(__1.z)))->x = (half)(((float2*)(&(v_.z)))->x);
    ((half2*)(&(__1.z)))->y = (half)(((float2*)(&(v_.z)))->y);
    ((half2*)(&(__1.w)))->x = (half)(((float2*)(&(v_.w)))->x);
    ((half2*)(&(__1.w)))->y = (half)(((float2*)(&(v_.w)))->y);
    *(uint4*)(compute + ((((((ax0_ax1_fused_0_256 * 32768) + (((int)threadIdx.z) * 16384)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8))) = __1;
  }
}


