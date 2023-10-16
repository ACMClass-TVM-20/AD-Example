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
__forceinline__ __device__ unsigned int
cast_smem_ptr_to_int(const void* const smem_ptr)
{
  unsigned int smem_int;
  asm volatile ("{ .reg .u64 smem_int; cvta.to.shared.u64 smem_int, %1; cvt.u32.u64 %0, smem_int; }"
    : "=r"(smem_int) : "l"(smem_ptr));
  return smem_int;
}

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
  float matmul_reindex_shared_dyn_warp[128];
  half A_reindex_shared_dyn_warp[32];
  half T_transpose_reindex_shared_dyn_warp[32];
  half A_reindex_shared_dyn_warp_1[32];
  half T_transpose_reindex_shared_dyn_warp_1[32];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 4; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 4; ++ax2_0_3_init) {
      for (int i = 0; i < 8; ++i) {
matmul_reindex_shared_dyn_warp[((ax1_0_3_init * 32) + (ax2_0_3_init * 8)) + i] = 0.0;}
;
    }
  }
  for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 4; ++ax0_ax1_fused_0_1) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_1 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + (((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_1 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int ax0_ax1_fused_0_2 = 0; ax0_ax1_fused_0_2 < 4; ++ax0_ax1_fused_0_2) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_2 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_2 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_3 = 0; ax0_ax1_fused_0_3 < 4; ++ax0_ax1_fused_0_3) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_3 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_3 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int ax0_ax1_fused_0_4 = 0; ax0_ax1_fused_0_4 < 4; ++ax0_ax1_fused_0_4) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_4 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_4 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 64))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_5 = 0; ax0_ax1_fused_0_5 < 4; ++ax0_ax1_fused_0_5) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_5 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_5 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 64))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int ax0_ax1_fused_0_6 = 0; ax0_ax1_fused_0_6 < 4; ++ax0_ax1_fused_0_6) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_6 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_6 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 96))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_7 = 0; ax0_ax1_fused_0_7 < 4; ++ax0_ax1_fused_0_7) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_7 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_7 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 96))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1 = 0; ax3_0_1 < 2; ++ax3_0_1) {
    for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0 * 512)) + (ax3_0_1 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_1 = 0; ax0_0_1 < 4; ++ax0_0_1) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_1 * 512)) + (ax3_0_1 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_1 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_1 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_1 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_1 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3 = 0; ax1_0_3 < 4; ++ax1_0_3) {
      for (int ax2_0_3 = 0; ax2_0_3 < 4; ++ax2_0_3) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_8 = 0; ax0_ax1_fused_0_8 < 4; ++ax0_ax1_fused_0_8) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_8 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_8 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 128))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_9 = 0; ax0_ax1_fused_0_9 < 4; ++ax0_ax1_fused_0_9) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_9 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_9 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 128))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 2; ++ax3_0_1_1) {
    for (int ax0_0_2 = 0; ax0_0_2 < 4; ++ax0_0_2) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_2 * 512)) + (ax3_0_1_1 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_2 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_2 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_2 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_2 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_3 = 0; ax0_0_3 < 4; ++ax0_0_3) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_3 * 512)) + (ax3_0_1_1 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_3 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_3 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_3 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_3 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 4; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 4; ++ax2_0_3_1) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_1 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_1 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_1 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_1 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_1 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_1 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_1 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_1 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_1 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_1 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_1 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_1 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_10 = 0; ax0_ax1_fused_0_10 < 4; ++ax0_ax1_fused_0_10) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_10 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_10 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 160))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_11 = 0; ax0_ax1_fused_0_11 < 4; ++ax0_ax1_fused_0_11) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_11 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_11 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 160))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_2 = 0; ax3_0_1_2 < 2; ++ax3_0_1_2) {
    for (int ax0_0_4 = 0; ax0_0_4 < 4; ++ax0_0_4) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_4 * 512)) + (ax3_0_1_2 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_4 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_4 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_4 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_4 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_5 = 0; ax0_0_5 < 4; ++ax0_0_5) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_5 * 512)) + (ax3_0_1_2 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_5 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_5 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_5 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_5 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 4; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 4; ++ax2_0_3_2) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_2 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_2 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_2 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_2 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_2 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_2 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_2 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_2 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_2 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_2 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_2 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_2 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_12 = 0; ax0_ax1_fused_0_12 < 4; ++ax0_ax1_fused_0_12) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_12 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_12 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 192))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_13 = 0; ax0_ax1_fused_0_13 < 4; ++ax0_ax1_fused_0_13) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_13 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_13 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 192))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_3 = 0; ax3_0_1_3 < 2; ++ax3_0_1_3) {
    for (int ax0_0_6 = 0; ax0_0_6 < 4; ++ax0_0_6) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_6 * 512)) + (ax3_0_1_3 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_6 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_6 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_6 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_6 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_7 = 0; ax0_0_7 < 4; ++ax0_0_7) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_7 * 512)) + (ax3_0_1_3 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_7 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_7 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_7 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_7 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_3 = 0; ax1_0_3_3 < 4; ++ax1_0_3_3) {
      for (int ax2_0_3_3 = 0; ax2_0_3_3 < 4; ++ax2_0_3_3) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_3 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_3 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_3 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_3 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_3 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_3 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_3 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_3 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_3 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_3 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_3 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_3 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_14 = 0; ax0_ax1_fused_0_14 < 4; ++ax0_ax1_fused_0_14) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_14 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_14 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 224))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_15 = 0; ax0_ax1_fused_0_15 < 4; ++ax0_ax1_fused_0_15) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_15 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_15 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 224))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_4 = 0; ax3_0_1_4 < 2; ++ax3_0_1_4) {
    for (int ax0_0_8 = 0; ax0_0_8 < 4; ++ax0_0_8) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_8 * 512)) + (ax3_0_1_4 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_8 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_8 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_8 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_8 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_9 = 0; ax0_0_9 < 4; ++ax0_0_9) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_9 * 512)) + (ax3_0_1_4 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_9 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_9 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_9 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_9 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_4 = 0; ax1_0_3_4 < 4; ++ax1_0_3_4) {
      for (int ax2_0_3_4 = 0; ax2_0_3_4 < 4; ++ax2_0_3_4) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_4 * 32) + (ax2_0_3_4 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_4 * 32) + (ax2_0_3_4 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_4 * 32) + (ax2_0_3_4 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_4 * 32) + (ax2_0_3_4 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_4 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_4 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_4 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_4 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_4 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_4 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_4 * 32) + (ax2_0_3_4 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_4 * 32) + (ax2_0_3_4 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_4 * 32) + (ax2_0_3_4 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_4 * 32) + (ax2_0_3_4 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_4 * 32) + (ax2_0_3_4 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_4 * 32) + (ax2_0_3_4 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_4 * 32) + (ax2_0_3_4 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_4 * 32) + (ax2_0_3_4 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_4 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_4 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_4 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_4 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_4 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_4 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_4 * 32) + (ax2_0_3_4 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_4 * 32) + (ax2_0_3_4 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_4 * 32) + (ax2_0_3_4 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_4 * 32) + (ax2_0_3_4 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_16 = 0; ax0_ax1_fused_0_16 < 4; ++ax0_ax1_fused_0_16) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_16 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_16 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 256))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_17 = 0; ax0_ax1_fused_0_17 < 4; ++ax0_ax1_fused_0_17) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_17 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_17 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 256))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_5 = 0; ax3_0_1_5 < 2; ++ax3_0_1_5) {
    for (int ax0_0_10 = 0; ax0_0_10 < 4; ++ax0_0_10) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_10 * 512)) + (ax3_0_1_5 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_10 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_10 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_10 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_10 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_11 = 0; ax0_0_11 < 4; ++ax0_0_11) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_11 * 512)) + (ax3_0_1_5 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_11 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_11 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_11 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_11 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_5 = 0; ax1_0_3_5 < 4; ++ax1_0_3_5) {
      for (int ax2_0_3_5 = 0; ax2_0_3_5 < 4; ++ax2_0_3_5) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_5 * 32) + (ax2_0_3_5 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_5 * 32) + (ax2_0_3_5 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_5 * 32) + (ax2_0_3_5 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_5 * 32) + (ax2_0_3_5 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_5 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_5 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_5 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_5 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_5 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_5 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_5 * 32) + (ax2_0_3_5 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_5 * 32) + (ax2_0_3_5 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_5 * 32) + (ax2_0_3_5 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_5 * 32) + (ax2_0_3_5 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_5 * 32) + (ax2_0_3_5 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_5 * 32) + (ax2_0_3_5 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_5 * 32) + (ax2_0_3_5 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_5 * 32) + (ax2_0_3_5 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_5 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_5 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_5 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_5 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_5 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_5 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_5 * 32) + (ax2_0_3_5 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_5 * 32) + (ax2_0_3_5 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_5 * 32) + (ax2_0_3_5 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_5 * 32) + (ax2_0_3_5 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_18 = 0; ax0_ax1_fused_0_18 < 4; ++ax0_ax1_fused_0_18) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_18 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_18 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 288))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_19 = 0; ax0_ax1_fused_0_19 < 4; ++ax0_ax1_fused_0_19) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_19 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_19 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 288))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_6 = 0; ax3_0_1_6 < 2; ++ax3_0_1_6) {
    for (int ax0_0_12 = 0; ax0_0_12 < 4; ++ax0_0_12) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_12 * 512)) + (ax3_0_1_6 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_12 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_12 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_12 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_12 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_13 = 0; ax0_0_13 < 4; ++ax0_0_13) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_13 * 512)) + (ax3_0_1_6 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_13 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_13 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_13 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_13 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_6 = 0; ax1_0_3_6 < 4; ++ax1_0_3_6) {
      for (int ax2_0_3_6 = 0; ax2_0_3_6 < 4; ++ax2_0_3_6) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_6 * 32) + (ax2_0_3_6 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_6 * 32) + (ax2_0_3_6 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_6 * 32) + (ax2_0_3_6 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_6 * 32) + (ax2_0_3_6 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_6 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_6 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_6 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_6 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_6 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_6 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_6 * 32) + (ax2_0_3_6 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_6 * 32) + (ax2_0_3_6 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_6 * 32) + (ax2_0_3_6 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_6 * 32) + (ax2_0_3_6 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_6 * 32) + (ax2_0_3_6 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_6 * 32) + (ax2_0_3_6 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_6 * 32) + (ax2_0_3_6 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_6 * 32) + (ax2_0_3_6 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_6 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_6 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_6 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_6 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_6 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_6 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_6 * 32) + (ax2_0_3_6 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_6 * 32) + (ax2_0_3_6 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_6 * 32) + (ax2_0_3_6 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_6 * 32) + (ax2_0_3_6 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_20 = 0; ax0_ax1_fused_0_20 < 4; ++ax0_ax1_fused_0_20) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_20 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_20 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 320))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_21 = 0; ax0_ax1_fused_0_21 < 4; ++ax0_ax1_fused_0_21) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_21 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_21 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 320))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_7 = 0; ax3_0_1_7 < 2; ++ax3_0_1_7) {
    for (int ax0_0_14 = 0; ax0_0_14 < 4; ++ax0_0_14) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_14 * 512)) + (ax3_0_1_7 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_14 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_14 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_14 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_14 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_15 = 0; ax0_0_15 < 4; ++ax0_0_15) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_15 * 512)) + (ax3_0_1_7 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_15 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_15 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_15 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_15 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_7 = 0; ax1_0_3_7 < 4; ++ax1_0_3_7) {
      for (int ax2_0_3_7 = 0; ax2_0_3_7 < 4; ++ax2_0_3_7) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_7 * 32) + (ax2_0_3_7 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_7 * 32) + (ax2_0_3_7 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_7 * 32) + (ax2_0_3_7 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_7 * 32) + (ax2_0_3_7 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_7 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_7 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_7 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_7 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_7 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_7 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_7 * 32) + (ax2_0_3_7 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_7 * 32) + (ax2_0_3_7 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_7 * 32) + (ax2_0_3_7 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_7 * 32) + (ax2_0_3_7 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_7 * 32) + (ax2_0_3_7 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_7 * 32) + (ax2_0_3_7 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_7 * 32) + (ax2_0_3_7 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_7 * 32) + (ax2_0_3_7 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_7 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_7 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_7 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_7 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_7 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_7 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_7 * 32) + (ax2_0_3_7 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_7 * 32) + (ax2_0_3_7 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_7 * 32) + (ax2_0_3_7 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_7 * 32) + (ax2_0_3_7 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_22 = 0; ax0_ax1_fused_0_22 < 4; ++ax0_ax1_fused_0_22) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_22 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_22 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 352))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_23 = 0; ax0_ax1_fused_0_23 < 4; ++ax0_ax1_fused_0_23) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_23 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_23 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 352))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_8 = 0; ax3_0_1_8 < 2; ++ax3_0_1_8) {
    for (int ax0_0_16 = 0; ax0_0_16 < 4; ++ax0_0_16) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_16 * 512)) + (ax3_0_1_8 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_16 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_16 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_16 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_16 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_17 = 0; ax0_0_17 < 4; ++ax0_0_17) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_17 * 512)) + (ax3_0_1_8 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_17 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_17 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_17 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_17 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_8 = 0; ax1_0_3_8 < 4; ++ax1_0_3_8) {
      for (int ax2_0_3_8 = 0; ax2_0_3_8 < 4; ++ax2_0_3_8) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_8 * 32) + (ax2_0_3_8 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_8 * 32) + (ax2_0_3_8 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_8 * 32) + (ax2_0_3_8 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_8 * 32) + (ax2_0_3_8 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_8 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_8 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_8 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_8 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_8 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_8 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_8 * 32) + (ax2_0_3_8 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_8 * 32) + (ax2_0_3_8 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_8 * 32) + (ax2_0_3_8 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_8 * 32) + (ax2_0_3_8 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_8 * 32) + (ax2_0_3_8 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_8 * 32) + (ax2_0_3_8 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_8 * 32) + (ax2_0_3_8 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_8 * 32) + (ax2_0_3_8 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_8 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_8 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_8 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_8 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_8 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_8 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_8 * 32) + (ax2_0_3_8 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_8 * 32) + (ax2_0_3_8 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_8 * 32) + (ax2_0_3_8 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_8 * 32) + (ax2_0_3_8 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_24 = 0; ax0_ax1_fused_0_24 < 4; ++ax0_ax1_fused_0_24) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_24 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_24 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 384))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_25 = 0; ax0_ax1_fused_0_25 < 4; ++ax0_ax1_fused_0_25) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_25 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_25 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 384))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_9 = 0; ax3_0_1_9 < 2; ++ax3_0_1_9) {
    for (int ax0_0_18 = 0; ax0_0_18 < 4; ++ax0_0_18) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_18 * 512)) + (ax3_0_1_9 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_18 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_18 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_18 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_18 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_19 = 0; ax0_0_19 < 4; ++ax0_0_19) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_19 * 512)) + (ax3_0_1_9 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_19 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_19 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_19 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_19 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_9 = 0; ax1_0_3_9 < 4; ++ax1_0_3_9) {
      for (int ax2_0_3_9 = 0; ax2_0_3_9 < 4; ++ax2_0_3_9) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_9 * 32) + (ax2_0_3_9 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_9 * 32) + (ax2_0_3_9 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_9 * 32) + (ax2_0_3_9 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_9 * 32) + (ax2_0_3_9 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_9 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_9 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_9 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_9 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_9 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_9 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_9 * 32) + (ax2_0_3_9 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_9 * 32) + (ax2_0_3_9 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_9 * 32) + (ax2_0_3_9 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_9 * 32) + (ax2_0_3_9 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_9 * 32) + (ax2_0_3_9 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_9 * 32) + (ax2_0_3_9 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_9 * 32) + (ax2_0_3_9 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_9 * 32) + (ax2_0_3_9 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_9 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_9 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_9 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_9 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_9 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_9 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_9 * 32) + (ax2_0_3_9 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_9 * 32) + (ax2_0_3_9 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_9 * 32) + (ax2_0_3_9 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_9 * 32) + (ax2_0_3_9 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_26 = 0; ax0_ax1_fused_0_26 < 4; ++ax0_ax1_fused_0_26) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_26 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_26 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 416))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_27 = 0; ax0_ax1_fused_0_27 < 4; ++ax0_ax1_fused_0_27) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_27 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_27 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 416))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_10 = 0; ax3_0_1_10 < 2; ++ax3_0_1_10) {
    for (int ax0_0_20 = 0; ax0_0_20 < 4; ++ax0_0_20) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_20 * 512)) + (ax3_0_1_10 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_20 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_20 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_20 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_20 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_21 = 0; ax0_0_21 < 4; ++ax0_0_21) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_21 * 512)) + (ax3_0_1_10 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_21 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_21 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_21 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_21 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_10 = 0; ax1_0_3_10 < 4; ++ax1_0_3_10) {
      for (int ax2_0_3_10 = 0; ax2_0_3_10 < 4; ++ax2_0_3_10) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_10 * 32) + (ax2_0_3_10 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_10 * 32) + (ax2_0_3_10 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_10 * 32) + (ax2_0_3_10 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_10 * 32) + (ax2_0_3_10 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_10 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_10 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_10 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_10 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_10 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_10 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_10 * 32) + (ax2_0_3_10 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_10 * 32) + (ax2_0_3_10 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_10 * 32) + (ax2_0_3_10 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_10 * 32) + (ax2_0_3_10 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_10 * 32) + (ax2_0_3_10 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_10 * 32) + (ax2_0_3_10 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_10 * 32) + (ax2_0_3_10 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_10 * 32) + (ax2_0_3_10 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_10 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_10 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_10 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_10 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_10 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_10 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_10 * 32) + (ax2_0_3_10 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_10 * 32) + (ax2_0_3_10 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_10 * 32) + (ax2_0_3_10 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_10 * 32) + (ax2_0_3_10 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_28 = 0; ax0_ax1_fused_0_28 < 4; ++ax0_ax1_fused_0_28) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_28 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_28 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 448))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_29 = 0; ax0_ax1_fused_0_29 < 4; ++ax0_ax1_fused_0_29) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_29 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_29 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 448))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_11 = 0; ax3_0_1_11 < 2; ++ax3_0_1_11) {
    for (int ax0_0_22 = 0; ax0_0_22 < 4; ++ax0_0_22) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_22 * 512)) + (ax3_0_1_11 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_22 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_22 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_22 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_22 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_23 = 0; ax0_0_23 < 4; ++ax0_0_23) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_23 * 512)) + (ax3_0_1_11 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_23 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_23 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_23 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_23 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_11 = 0; ax1_0_3_11 < 4; ++ax1_0_3_11) {
      for (int ax2_0_3_11 = 0; ax2_0_3_11 < 4; ++ax2_0_3_11) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_11 * 32) + (ax2_0_3_11 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_11 * 32) + (ax2_0_3_11 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_11 * 32) + (ax2_0_3_11 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_11 * 32) + (ax2_0_3_11 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_11 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_11 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_11 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_11 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_11 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_11 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_11 * 32) + (ax2_0_3_11 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_11 * 32) + (ax2_0_3_11 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_11 * 32) + (ax2_0_3_11 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_11 * 32) + (ax2_0_3_11 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_11 * 32) + (ax2_0_3_11 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_11 * 32) + (ax2_0_3_11 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_11 * 32) + (ax2_0_3_11 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_11 * 32) + (ax2_0_3_11 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_11 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_11 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_11 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_11 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_11 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_11 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_11 * 32) + (ax2_0_3_11 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_11 * 32) + (ax2_0_3_11 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_11 * 32) + (ax2_0_3_11 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_11 * 32) + (ax2_0_3_11 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_30 = 0; ax0_ax1_fused_0_30 < 4; ++ax0_ax1_fused_0_30) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_30 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_30 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 480))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_31 = 0; ax0_ax1_fused_0_31 < 4; ++ax0_ax1_fused_0_31) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_31 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_31 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 480))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_12 = 0; ax3_0_1_12 < 2; ++ax3_0_1_12) {
    for (int ax0_0_24 = 0; ax0_0_24 < 4; ++ax0_0_24) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_24 * 512)) + (ax3_0_1_12 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_24 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_24 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_24 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_24 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_25 = 0; ax0_0_25 < 4; ++ax0_0_25) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_25 * 512)) + (ax3_0_1_12 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_25 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_25 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_25 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_25 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_12 = 0; ax1_0_3_12 < 4; ++ax1_0_3_12) {
      for (int ax2_0_3_12 = 0; ax2_0_3_12 < 4; ++ax2_0_3_12) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_12 * 32) + (ax2_0_3_12 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_12 * 32) + (ax2_0_3_12 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_12 * 32) + (ax2_0_3_12 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_12 * 32) + (ax2_0_3_12 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_12 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_12 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_12 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_12 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_12 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_12 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_12 * 32) + (ax2_0_3_12 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_12 * 32) + (ax2_0_3_12 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_12 * 32) + (ax2_0_3_12 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_12 * 32) + (ax2_0_3_12 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_12 * 32) + (ax2_0_3_12 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_12 * 32) + (ax2_0_3_12 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_12 * 32) + (ax2_0_3_12 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_12 * 32) + (ax2_0_3_12 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_12 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_12 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_12 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_12 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_12 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_12 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_12 * 32) + (ax2_0_3_12 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_12 * 32) + (ax2_0_3_12 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_12 * 32) + (ax2_0_3_12 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_12 * 32) + (ax2_0_3_12 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_32 = 0; ax0_ax1_fused_0_32 < 4; ++ax0_ax1_fused_0_32) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_32 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_32 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 512))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_33 = 0; ax0_ax1_fused_0_33 < 4; ++ax0_ax1_fused_0_33) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_33 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_33 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 512))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_13 = 0; ax3_0_1_13 < 2; ++ax3_0_1_13) {
    for (int ax0_0_26 = 0; ax0_0_26 < 4; ++ax0_0_26) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_26 * 512)) + (ax3_0_1_13 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_26 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_26 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_26 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_26 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_27 = 0; ax0_0_27 < 4; ++ax0_0_27) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_27 * 512)) + (ax3_0_1_13 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_27 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_27 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_27 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_27 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_13 = 0; ax1_0_3_13 < 4; ++ax1_0_3_13) {
      for (int ax2_0_3_13 = 0; ax2_0_3_13 < 4; ++ax2_0_3_13) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_13 * 32) + (ax2_0_3_13 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_13 * 32) + (ax2_0_3_13 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_13 * 32) + (ax2_0_3_13 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_13 * 32) + (ax2_0_3_13 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_13 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_13 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_13 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_13 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_13 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_13 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_13 * 32) + (ax2_0_3_13 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_13 * 32) + (ax2_0_3_13 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_13 * 32) + (ax2_0_3_13 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_13 * 32) + (ax2_0_3_13 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_13 * 32) + (ax2_0_3_13 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_13 * 32) + (ax2_0_3_13 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_13 * 32) + (ax2_0_3_13 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_13 * 32) + (ax2_0_3_13 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_13 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_13 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_13 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_13 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_13 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_13 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_13 * 32) + (ax2_0_3_13 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_13 * 32) + (ax2_0_3_13 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_13 * 32) + (ax2_0_3_13 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_13 * 32) + (ax2_0_3_13 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_34 = 0; ax0_ax1_fused_0_34 < 4; ++ax0_ax1_fused_0_34) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_34 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_34 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 544))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_35 = 0; ax0_ax1_fused_0_35 < 4; ++ax0_ax1_fused_0_35) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_35 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_35 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 544))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_14 = 0; ax3_0_1_14 < 2; ++ax3_0_1_14) {
    for (int ax0_0_28 = 0; ax0_0_28 < 4; ++ax0_0_28) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_28 * 512)) + (ax3_0_1_14 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_28 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_28 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_28 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_28 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_29 = 0; ax0_0_29 < 4; ++ax0_0_29) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_29 * 512)) + (ax3_0_1_14 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_29 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_29 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_29 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_29 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_14 = 0; ax1_0_3_14 < 4; ++ax1_0_3_14) {
      for (int ax2_0_3_14 = 0; ax2_0_3_14 < 4; ++ax2_0_3_14) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_14 * 32) + (ax2_0_3_14 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_14 * 32) + (ax2_0_3_14 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_14 * 32) + (ax2_0_3_14 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_14 * 32) + (ax2_0_3_14 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_14 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_14 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_14 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_14 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_14 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_14 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_14 * 32) + (ax2_0_3_14 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_14 * 32) + (ax2_0_3_14 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_14 * 32) + (ax2_0_3_14 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_14 * 32) + (ax2_0_3_14 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_14 * 32) + (ax2_0_3_14 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_14 * 32) + (ax2_0_3_14 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_14 * 32) + (ax2_0_3_14 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_14 * 32) + (ax2_0_3_14 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_14 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_14 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_14 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_14 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_14 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_14 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_14 * 32) + (ax2_0_3_14 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_14 * 32) + (ax2_0_3_14 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_14 * 32) + (ax2_0_3_14 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_14 * 32) + (ax2_0_3_14 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_36 = 0; ax0_ax1_fused_0_36 < 4; ++ax0_ax1_fused_0_36) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_36 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_36 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 576))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_37 = 0; ax0_ax1_fused_0_37 < 4; ++ax0_ax1_fused_0_37) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_37 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_37 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 576))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_15 = 0; ax3_0_1_15 < 2; ++ax3_0_1_15) {
    for (int ax0_0_30 = 0; ax0_0_30 < 4; ++ax0_0_30) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_30 * 512)) + (ax3_0_1_15 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_30 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_30 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_30 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_30 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_31 = 0; ax0_0_31 < 4; ++ax0_0_31) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_31 * 512)) + (ax3_0_1_15 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_31 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_31 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_31 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_31 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_15 = 0; ax1_0_3_15 < 4; ++ax1_0_3_15) {
      for (int ax2_0_3_15 = 0; ax2_0_3_15 < 4; ++ax2_0_3_15) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_15 * 32) + (ax2_0_3_15 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_15 * 32) + (ax2_0_3_15 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_15 * 32) + (ax2_0_3_15 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_15 * 32) + (ax2_0_3_15 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_15 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_15 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_15 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_15 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_15 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_15 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_15 * 32) + (ax2_0_3_15 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_15 * 32) + (ax2_0_3_15 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_15 * 32) + (ax2_0_3_15 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_15 * 32) + (ax2_0_3_15 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_15 * 32) + (ax2_0_3_15 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_15 * 32) + (ax2_0_3_15 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_15 * 32) + (ax2_0_3_15 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_15 * 32) + (ax2_0_3_15 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_15 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_15 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_15 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_15 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_15 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_15 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_15 * 32) + (ax2_0_3_15 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_15 * 32) + (ax2_0_3_15 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_15 * 32) + (ax2_0_3_15 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_15 * 32) + (ax2_0_3_15 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_38 = 0; ax0_ax1_fused_0_38 < 4; ++ax0_ax1_fused_0_38) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_38 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_38 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 608))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_39 = 0; ax0_ax1_fused_0_39 < 4; ++ax0_ax1_fused_0_39) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_39 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_39 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 608))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_16 = 0; ax3_0_1_16 < 2; ++ax3_0_1_16) {
    for (int ax0_0_32 = 0; ax0_0_32 < 4; ++ax0_0_32) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_32 * 512)) + (ax3_0_1_16 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_32 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_32 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_32 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_32 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_33 = 0; ax0_0_33 < 4; ++ax0_0_33) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_33 * 512)) + (ax3_0_1_16 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_33 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_33 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_33 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_33 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_16 = 0; ax1_0_3_16 < 4; ++ax1_0_3_16) {
      for (int ax2_0_3_16 = 0; ax2_0_3_16 < 4; ++ax2_0_3_16) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_16 * 32) + (ax2_0_3_16 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_16 * 32) + (ax2_0_3_16 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_16 * 32) + (ax2_0_3_16 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_16 * 32) + (ax2_0_3_16 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_16 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_16 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_16 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_16 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_16 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_16 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_16 * 32) + (ax2_0_3_16 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_16 * 32) + (ax2_0_3_16 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_16 * 32) + (ax2_0_3_16 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_16 * 32) + (ax2_0_3_16 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_16 * 32) + (ax2_0_3_16 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_16 * 32) + (ax2_0_3_16 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_16 * 32) + (ax2_0_3_16 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_16 * 32) + (ax2_0_3_16 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_16 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_16 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_16 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_16 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_16 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_16 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_16 * 32) + (ax2_0_3_16 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_16 * 32) + (ax2_0_3_16 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_16 * 32) + (ax2_0_3_16 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_16 * 32) + (ax2_0_3_16 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_40 = 0; ax0_ax1_fused_0_40 < 4; ++ax0_ax1_fused_0_40) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_40 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_40 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 640))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_41 = 0; ax0_ax1_fused_0_41 < 4; ++ax0_ax1_fused_0_41) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_41 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_41 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 640))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_17 = 0; ax3_0_1_17 < 2; ++ax3_0_1_17) {
    for (int ax0_0_34 = 0; ax0_0_34 < 4; ++ax0_0_34) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_34 * 512)) + (ax3_0_1_17 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_34 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_34 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_34 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_34 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_35 = 0; ax0_0_35 < 4; ++ax0_0_35) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_35 * 512)) + (ax3_0_1_17 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_35 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_35 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_35 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_35 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_17 = 0; ax1_0_3_17 < 4; ++ax1_0_3_17) {
      for (int ax2_0_3_17 = 0; ax2_0_3_17 < 4; ++ax2_0_3_17) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_17 * 32) + (ax2_0_3_17 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_17 * 32) + (ax2_0_3_17 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_17 * 32) + (ax2_0_3_17 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_17 * 32) + (ax2_0_3_17 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_17 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_17 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_17 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_17 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_17 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_17 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_17 * 32) + (ax2_0_3_17 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_17 * 32) + (ax2_0_3_17 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_17 * 32) + (ax2_0_3_17 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_17 * 32) + (ax2_0_3_17 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_17 * 32) + (ax2_0_3_17 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_17 * 32) + (ax2_0_3_17 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_17 * 32) + (ax2_0_3_17 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_17 * 32) + (ax2_0_3_17 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_17 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_17 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_17 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_17 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_17 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_17 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_17 * 32) + (ax2_0_3_17 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_17 * 32) + (ax2_0_3_17 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_17 * 32) + (ax2_0_3_17 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_17 * 32) + (ax2_0_3_17 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_42 = 0; ax0_ax1_fused_0_42 < 4; ++ax0_ax1_fused_0_42) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_42 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_42 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 672))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_43 = 0; ax0_ax1_fused_0_43 < 4; ++ax0_ax1_fused_0_43) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_43 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_43 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 672))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_18 = 0; ax3_0_1_18 < 2; ++ax3_0_1_18) {
    for (int ax0_0_36 = 0; ax0_0_36 < 4; ++ax0_0_36) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_36 * 512)) + (ax3_0_1_18 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_36 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_36 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_36 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_36 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_37 = 0; ax0_0_37 < 4; ++ax0_0_37) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_37 * 512)) + (ax3_0_1_18 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_37 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_37 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_37 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_37 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_18 = 0; ax1_0_3_18 < 4; ++ax1_0_3_18) {
      for (int ax2_0_3_18 = 0; ax2_0_3_18 < 4; ++ax2_0_3_18) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_18 * 32) + (ax2_0_3_18 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_18 * 32) + (ax2_0_3_18 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_18 * 32) + (ax2_0_3_18 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_18 * 32) + (ax2_0_3_18 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_18 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_18 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_18 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_18 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_18 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_18 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_18 * 32) + (ax2_0_3_18 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_18 * 32) + (ax2_0_3_18 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_18 * 32) + (ax2_0_3_18 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_18 * 32) + (ax2_0_3_18 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_18 * 32) + (ax2_0_3_18 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_18 * 32) + (ax2_0_3_18 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_18 * 32) + (ax2_0_3_18 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_18 * 32) + (ax2_0_3_18 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_18 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_18 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_18 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_18 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_18 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_18 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_18 * 32) + (ax2_0_3_18 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_18 * 32) + (ax2_0_3_18 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_18 * 32) + (ax2_0_3_18 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_18 * 32) + (ax2_0_3_18 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_44 = 0; ax0_ax1_fused_0_44 < 4; ++ax0_ax1_fused_0_44) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_44 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_44 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 704))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_45 = 0; ax0_ax1_fused_0_45 < 4; ++ax0_ax1_fused_0_45) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_45 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_45 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 704))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_19 = 0; ax3_0_1_19 < 2; ++ax3_0_1_19) {
    for (int ax0_0_38 = 0; ax0_0_38 < 4; ++ax0_0_38) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_38 * 512)) + (ax3_0_1_19 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_38 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_38 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_38 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_38 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_39 = 0; ax0_0_39 < 4; ++ax0_0_39) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_39 * 512)) + (ax3_0_1_19 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_39 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_39 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_39 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_39 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_19 = 0; ax1_0_3_19 < 4; ++ax1_0_3_19) {
      for (int ax2_0_3_19 = 0; ax2_0_3_19 < 4; ++ax2_0_3_19) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_19 * 32) + (ax2_0_3_19 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_19 * 32) + (ax2_0_3_19 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_19 * 32) + (ax2_0_3_19 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_19 * 32) + (ax2_0_3_19 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_19 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_19 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_19 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_19 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_19 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_19 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_19 * 32) + (ax2_0_3_19 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_19 * 32) + (ax2_0_3_19 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_19 * 32) + (ax2_0_3_19 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_19 * 32) + (ax2_0_3_19 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_19 * 32) + (ax2_0_3_19 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_19 * 32) + (ax2_0_3_19 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_19 * 32) + (ax2_0_3_19 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_19 * 32) + (ax2_0_3_19 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_19 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_19 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_19 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_19 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_19 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_19 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_19 * 32) + (ax2_0_3_19 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_19 * 32) + (ax2_0_3_19 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_19 * 32) + (ax2_0_3_19 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_19 * 32) + (ax2_0_3_19 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_46 = 0; ax0_ax1_fused_0_46 < 4; ++ax0_ax1_fused_0_46) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_46 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_46 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 736))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_47 = 0; ax0_ax1_fused_0_47 < 4; ++ax0_ax1_fused_0_47) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_47 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_47 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 736))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_20 = 0; ax3_0_1_20 < 2; ++ax3_0_1_20) {
    for (int ax0_0_40 = 0; ax0_0_40 < 4; ++ax0_0_40) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_40 * 512)) + (ax3_0_1_20 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_40 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_40 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_40 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_40 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_41 = 0; ax0_0_41 < 4; ++ax0_0_41) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_41 * 512)) + (ax3_0_1_20 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_41 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_41 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_41 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_41 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_20 = 0; ax1_0_3_20 < 4; ++ax1_0_3_20) {
      for (int ax2_0_3_20 = 0; ax2_0_3_20 < 4; ++ax2_0_3_20) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_20 * 32) + (ax2_0_3_20 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_20 * 32) + (ax2_0_3_20 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_20 * 32) + (ax2_0_3_20 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_20 * 32) + (ax2_0_3_20 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_20 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_20 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_20 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_20 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_20 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_20 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_20 * 32) + (ax2_0_3_20 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_20 * 32) + (ax2_0_3_20 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_20 * 32) + (ax2_0_3_20 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_20 * 32) + (ax2_0_3_20 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_20 * 32) + (ax2_0_3_20 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_20 * 32) + (ax2_0_3_20 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_20 * 32) + (ax2_0_3_20 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_20 * 32) + (ax2_0_3_20 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_20 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_20 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_20 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_20 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_20 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_20 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_20 * 32) + (ax2_0_3_20 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_20 * 32) + (ax2_0_3_20 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_20 * 32) + (ax2_0_3_20 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_20 * 32) + (ax2_0_3_20 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_48 = 0; ax0_ax1_fused_0_48 < 4; ++ax0_ax1_fused_0_48) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_48 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_48 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 768))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_49 = 0; ax0_ax1_fused_0_49 < 4; ++ax0_ax1_fused_0_49) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_49 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_49 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 768))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_21 = 0; ax3_0_1_21 < 2; ++ax3_0_1_21) {
    for (int ax0_0_42 = 0; ax0_0_42 < 4; ++ax0_0_42) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_42 * 512)) + (ax3_0_1_21 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_42 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_42 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_42 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_42 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_43 = 0; ax0_0_43 < 4; ++ax0_0_43) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_43 * 512)) + (ax3_0_1_21 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_43 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_43 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_43 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_43 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_21 = 0; ax1_0_3_21 < 4; ++ax1_0_3_21) {
      for (int ax2_0_3_21 = 0; ax2_0_3_21 < 4; ++ax2_0_3_21) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_21 * 32) + (ax2_0_3_21 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_21 * 32) + (ax2_0_3_21 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_21 * 32) + (ax2_0_3_21 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_21 * 32) + (ax2_0_3_21 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_21 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_21 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_21 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_21 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_21 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_21 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_21 * 32) + (ax2_0_3_21 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_21 * 32) + (ax2_0_3_21 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_21 * 32) + (ax2_0_3_21 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_21 * 32) + (ax2_0_3_21 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_21 * 32) + (ax2_0_3_21 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_21 * 32) + (ax2_0_3_21 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_21 * 32) + (ax2_0_3_21 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_21 * 32) + (ax2_0_3_21 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_21 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_21 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_21 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_21 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_21 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_21 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_21 * 32) + (ax2_0_3_21 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_21 * 32) + (ax2_0_3_21 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_21 * 32) + (ax2_0_3_21 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_21 * 32) + (ax2_0_3_21 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_50 = 0; ax0_ax1_fused_0_50 < 4; ++ax0_ax1_fused_0_50) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_50 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_50 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 800))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_51 = 0; ax0_ax1_fused_0_51 < 4; ++ax0_ax1_fused_0_51) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_51 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_51 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 800))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_22 = 0; ax3_0_1_22 < 2; ++ax3_0_1_22) {
    for (int ax0_0_44 = 0; ax0_0_44 < 4; ++ax0_0_44) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_44 * 512)) + (ax3_0_1_22 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_44 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_44 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_44 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_44 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_45 = 0; ax0_0_45 < 4; ++ax0_0_45) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_45 * 512)) + (ax3_0_1_22 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_45 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_45 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_45 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_45 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_22 = 0; ax1_0_3_22 < 4; ++ax1_0_3_22) {
      for (int ax2_0_3_22 = 0; ax2_0_3_22 < 4; ++ax2_0_3_22) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_22 * 32) + (ax2_0_3_22 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_22 * 32) + (ax2_0_3_22 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_22 * 32) + (ax2_0_3_22 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_22 * 32) + (ax2_0_3_22 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_22 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_22 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_22 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_22 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_22 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_22 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_22 * 32) + (ax2_0_3_22 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_22 * 32) + (ax2_0_3_22 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_22 * 32) + (ax2_0_3_22 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_22 * 32) + (ax2_0_3_22 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_22 * 32) + (ax2_0_3_22 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_22 * 32) + (ax2_0_3_22 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_22 * 32) + (ax2_0_3_22 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_22 * 32) + (ax2_0_3_22 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_22 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_22 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_22 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_22 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_22 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_22 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_22 * 32) + (ax2_0_3_22 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_22 * 32) + (ax2_0_3_22 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_22 * 32) + (ax2_0_3_22 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_22 * 32) + (ax2_0_3_22 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_52 = 0; ax0_ax1_fused_0_52 < 4; ++ax0_ax1_fused_0_52) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_52 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_52 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 832))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_53 = 0; ax0_ax1_fused_0_53 < 4; ++ax0_ax1_fused_0_53) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_53 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_53 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 832))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_23 = 0; ax3_0_1_23 < 2; ++ax3_0_1_23) {
    for (int ax0_0_46 = 0; ax0_0_46 < 4; ++ax0_0_46) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_46 * 512)) + (ax3_0_1_23 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_46 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_46 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_46 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_46 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_47 = 0; ax0_0_47 < 4; ++ax0_0_47) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_47 * 512)) + (ax3_0_1_23 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_47 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_47 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_47 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_47 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_23 = 0; ax1_0_3_23 < 4; ++ax1_0_3_23) {
      for (int ax2_0_3_23 = 0; ax2_0_3_23 < 4; ++ax2_0_3_23) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_23 * 32) + (ax2_0_3_23 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_23 * 32) + (ax2_0_3_23 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_23 * 32) + (ax2_0_3_23 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_23 * 32) + (ax2_0_3_23 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_23 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_23 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_23 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_23 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_23 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_23 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_23 * 32) + (ax2_0_3_23 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_23 * 32) + (ax2_0_3_23 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_23 * 32) + (ax2_0_3_23 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_23 * 32) + (ax2_0_3_23 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_23 * 32) + (ax2_0_3_23 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_23 * 32) + (ax2_0_3_23 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_23 * 32) + (ax2_0_3_23 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_23 * 32) + (ax2_0_3_23 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_23 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_23 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_23 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_23 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_23 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_23 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_23 * 32) + (ax2_0_3_23 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_23 * 32) + (ax2_0_3_23 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_23 * 32) + (ax2_0_3_23 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_23 * 32) + (ax2_0_3_23 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_54 = 0; ax0_ax1_fused_0_54 < 4; ++ax0_ax1_fused_0_54) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_54 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_54 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 864))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_55 = 0; ax0_ax1_fused_0_55 < 4; ++ax0_ax1_fused_0_55) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_55 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_55 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 864))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_24 = 0; ax3_0_1_24 < 2; ++ax3_0_1_24) {
    for (int ax0_0_48 = 0; ax0_0_48 < 4; ++ax0_0_48) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_48 * 512)) + (ax3_0_1_24 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_48 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_48 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_48 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_48 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_49 = 0; ax0_0_49 < 4; ++ax0_0_49) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_49 * 512)) + (ax3_0_1_24 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_49 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_49 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_49 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_49 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_24 = 0; ax1_0_3_24 < 4; ++ax1_0_3_24) {
      for (int ax2_0_3_24 = 0; ax2_0_3_24 < 4; ++ax2_0_3_24) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_24 * 32) + (ax2_0_3_24 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_24 * 32) + (ax2_0_3_24 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_24 * 32) + (ax2_0_3_24 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_24 * 32) + (ax2_0_3_24 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_24 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_24 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_24 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_24 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_24 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_24 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_24 * 32) + (ax2_0_3_24 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_24 * 32) + (ax2_0_3_24 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_24 * 32) + (ax2_0_3_24 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_24 * 32) + (ax2_0_3_24 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_24 * 32) + (ax2_0_3_24 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_24 * 32) + (ax2_0_3_24 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_24 * 32) + (ax2_0_3_24 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_24 * 32) + (ax2_0_3_24 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_24 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_24 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_24 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_24 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_24 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_24 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_24 * 32) + (ax2_0_3_24 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_24 * 32) + (ax2_0_3_24 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_24 * 32) + (ax2_0_3_24 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_24 * 32) + (ax2_0_3_24 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_56 = 0; ax0_ax1_fused_0_56 < 4; ++ax0_ax1_fused_0_56) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_56 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_56 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 896))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_57 = 0; ax0_ax1_fused_0_57 < 4; ++ax0_ax1_fused_0_57) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_57 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_57 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 896))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_25 = 0; ax3_0_1_25 < 2; ++ax3_0_1_25) {
    for (int ax0_0_50 = 0; ax0_0_50 < 4; ++ax0_0_50) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_50 * 512)) + (ax3_0_1_25 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_50 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_50 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_50 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_50 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_51 = 0; ax0_0_51 < 4; ++ax0_0_51) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_51 * 512)) + (ax3_0_1_25 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_51 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_51 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_51 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_51 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_25 = 0; ax1_0_3_25 < 4; ++ax1_0_3_25) {
      for (int ax2_0_3_25 = 0; ax2_0_3_25 < 4; ++ax2_0_3_25) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_25 * 32) + (ax2_0_3_25 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_25 * 32) + (ax2_0_3_25 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_25 * 32) + (ax2_0_3_25 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_25 * 32) + (ax2_0_3_25 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_25 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_25 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_25 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_25 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_25 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_25 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_25 * 32) + (ax2_0_3_25 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_25 * 32) + (ax2_0_3_25 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_25 * 32) + (ax2_0_3_25 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_25 * 32) + (ax2_0_3_25 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_25 * 32) + (ax2_0_3_25 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_25 * 32) + (ax2_0_3_25 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_25 * 32) + (ax2_0_3_25 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_25 * 32) + (ax2_0_3_25 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_25 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_25 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_25 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_25 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_25 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_25 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_25 * 32) + (ax2_0_3_25 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_25 * 32) + (ax2_0_3_25 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_25 * 32) + (ax2_0_3_25 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_25 * 32) + (ax2_0_3_25 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_58 = 0; ax0_ax1_fused_0_58 < 4; ++ax0_ax1_fused_0_58) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_58 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_58 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 928))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_59 = 0; ax0_ax1_fused_0_59 < 4; ++ax0_ax1_fused_0_59) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_59 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_59 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 928))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_26 = 0; ax3_0_1_26 < 2; ++ax3_0_1_26) {
    for (int ax0_0_52 = 0; ax0_0_52 < 4; ++ax0_0_52) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_52 * 512)) + (ax3_0_1_26 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_52 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_52 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_52 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_52 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_53 = 0; ax0_0_53 < 4; ++ax0_0_53) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_53 * 512)) + (ax3_0_1_26 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_53 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_53 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_53 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_53 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_26 = 0; ax1_0_3_26 < 4; ++ax1_0_3_26) {
      for (int ax2_0_3_26 = 0; ax2_0_3_26 < 4; ++ax2_0_3_26) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_26 * 32) + (ax2_0_3_26 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_26 * 32) + (ax2_0_3_26 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_26 * 32) + (ax2_0_3_26 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_26 * 32) + (ax2_0_3_26 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_26 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_26 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_26 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_26 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_26 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_26 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_26 * 32) + (ax2_0_3_26 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_26 * 32) + (ax2_0_3_26 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_26 * 32) + (ax2_0_3_26 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_26 * 32) + (ax2_0_3_26 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_26 * 32) + (ax2_0_3_26 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_26 * 32) + (ax2_0_3_26 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_26 * 32) + (ax2_0_3_26 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_26 * 32) + (ax2_0_3_26 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_26 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_26 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_26 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_26 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_26 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_26 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_26 * 32) + (ax2_0_3_26 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_26 * 32) + (ax2_0_3_26 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_26 * 32) + (ax2_0_3_26 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_26 * 32) + (ax2_0_3_26 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_60 = 0; ax0_ax1_fused_0_60 < 4; ++ax0_ax1_fused_0_60) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_60 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_60 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 960))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_61 = 0; ax0_ax1_fused_0_61 < 4; ++ax0_ax1_fused_0_61) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_61 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_61 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 960))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_27 = 0; ax3_0_1_27 < 2; ++ax3_0_1_27) {
    for (int ax0_0_54 = 0; ax0_0_54 < 4; ++ax0_0_54) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_54 * 512)) + (ax3_0_1_27 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_54 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_54 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_54 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_54 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_55 = 0; ax0_0_55 < 4; ++ax0_0_55) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_55 * 512)) + (ax3_0_1_27 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_55 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_55 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_55 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_55 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_27 = 0; ax1_0_3_27 < 4; ++ax1_0_3_27) {
      for (int ax2_0_3_27 = 0; ax2_0_3_27 < 4; ++ax2_0_3_27) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_27 * 32) + (ax2_0_3_27 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_27 * 32) + (ax2_0_3_27 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_27 * 32) + (ax2_0_3_27 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_27 * 32) + (ax2_0_3_27 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_27 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_27 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_27 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_27 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_27 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_27 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_27 * 32) + (ax2_0_3_27 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_27 * 32) + (ax2_0_3_27 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_27 * 32) + (ax2_0_3_27 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_27 * 32) + (ax2_0_3_27 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_27 * 32) + (ax2_0_3_27 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_27 * 32) + (ax2_0_3_27 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_27 * 32) + (ax2_0_3_27 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_27 * 32) + (ax2_0_3_27 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_27 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_27 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_27 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_27 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_27 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_27 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_27 * 32) + (ax2_0_3_27 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_27 * 32) + (ax2_0_3_27 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_27 * 32) + (ax2_0_3_27 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_27 * 32) + (ax2_0_3_27 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_62 = 0; ax0_ax1_fused_0_62 < 4; ++ax0_ax1_fused_0_62) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_62 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_62 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 992))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_63 = 0; ax0_ax1_fused_0_63 < 4; ++ax0_ax1_fused_0_63) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_63 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_63 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 992))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_28 = 0; ax3_0_1_28 < 2; ++ax3_0_1_28) {
    for (int ax0_0_56 = 0; ax0_0_56 < 4; ++ax0_0_56) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_56 * 512)) + (ax3_0_1_28 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_56 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_56 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_56 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_56 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_57 = 0; ax0_0_57 < 4; ++ax0_0_57) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_57 * 512)) + (ax3_0_1_28 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_57 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_57 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_57 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_57 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_28 = 0; ax1_0_3_28 < 4; ++ax1_0_3_28) {
      for (int ax2_0_3_28 = 0; ax2_0_3_28 < 4; ++ax2_0_3_28) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_28 * 32) + (ax2_0_3_28 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_28 * 32) + (ax2_0_3_28 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_28 * 32) + (ax2_0_3_28 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_28 * 32) + (ax2_0_3_28 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_28 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_28 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_28 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_28 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_28 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_28 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_28 * 32) + (ax2_0_3_28 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_28 * 32) + (ax2_0_3_28 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_28 * 32) + (ax2_0_3_28 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_28 * 32) + (ax2_0_3_28 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_28 * 32) + (ax2_0_3_28 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_28 * 32) + (ax2_0_3_28 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_28 * 32) + (ax2_0_3_28 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_28 * 32) + (ax2_0_3_28 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_28 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_28 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_28 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_28 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_28 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_28 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_28 * 32) + (ax2_0_3_28 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_28 * 32) + (ax2_0_3_28 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_28 * 32) + (ax2_0_3_28 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_28 * 32) + (ax2_0_3_28 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_64 = 0; ax0_ax1_fused_0_64 < 4; ++ax0_ax1_fused_0_64) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_64 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_64 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1024))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_65 = 0; ax0_ax1_fused_0_65 < 4; ++ax0_ax1_fused_0_65) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_65 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_65 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1024))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_29 = 0; ax3_0_1_29 < 2; ++ax3_0_1_29) {
    for (int ax0_0_58 = 0; ax0_0_58 < 4; ++ax0_0_58) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_58 * 512)) + (ax3_0_1_29 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_58 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_58 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_58 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_58 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_59 = 0; ax0_0_59 < 4; ++ax0_0_59) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_59 * 512)) + (ax3_0_1_29 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_59 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_59 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_59 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_59 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_29 = 0; ax1_0_3_29 < 4; ++ax1_0_3_29) {
      for (int ax2_0_3_29 = 0; ax2_0_3_29 < 4; ++ax2_0_3_29) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_29 * 32) + (ax2_0_3_29 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_29 * 32) + (ax2_0_3_29 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_29 * 32) + (ax2_0_3_29 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_29 * 32) + (ax2_0_3_29 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_29 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_29 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_29 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_29 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_29 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_29 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_29 * 32) + (ax2_0_3_29 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_29 * 32) + (ax2_0_3_29 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_29 * 32) + (ax2_0_3_29 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_29 * 32) + (ax2_0_3_29 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_29 * 32) + (ax2_0_3_29 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_29 * 32) + (ax2_0_3_29 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_29 * 32) + (ax2_0_3_29 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_29 * 32) + (ax2_0_3_29 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_29 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_29 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_29 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_29 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_29 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_29 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_29 * 32) + (ax2_0_3_29 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_29 * 32) + (ax2_0_3_29 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_29 * 32) + (ax2_0_3_29 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_29 * 32) + (ax2_0_3_29 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_66 = 0; ax0_ax1_fused_0_66 < 4; ++ax0_ax1_fused_0_66) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_66 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_66 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1056))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_67 = 0; ax0_ax1_fused_0_67 < 4; ++ax0_ax1_fused_0_67) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_67 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_67 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1056))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_30 = 0; ax3_0_1_30 < 2; ++ax3_0_1_30) {
    for (int ax0_0_60 = 0; ax0_0_60 < 4; ++ax0_0_60) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_60 * 512)) + (ax3_0_1_30 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_60 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_60 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_60 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_60 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_61 = 0; ax0_0_61 < 4; ++ax0_0_61) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_61 * 512)) + (ax3_0_1_30 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_61 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_61 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_61 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_61 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_30 = 0; ax1_0_3_30 < 4; ++ax1_0_3_30) {
      for (int ax2_0_3_30 = 0; ax2_0_3_30 < 4; ++ax2_0_3_30) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_30 * 32) + (ax2_0_3_30 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_30 * 32) + (ax2_0_3_30 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_30 * 32) + (ax2_0_3_30 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_30 * 32) + (ax2_0_3_30 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_30 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_30 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_30 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_30 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_30 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_30 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_30 * 32) + (ax2_0_3_30 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_30 * 32) + (ax2_0_3_30 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_30 * 32) + (ax2_0_3_30 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_30 * 32) + (ax2_0_3_30 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_30 * 32) + (ax2_0_3_30 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_30 * 32) + (ax2_0_3_30 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_30 * 32) + (ax2_0_3_30 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_30 * 32) + (ax2_0_3_30 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_30 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_30 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_30 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_30 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_30 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_30 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_30 * 32) + (ax2_0_3_30 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_30 * 32) + (ax2_0_3_30 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_30 * 32) + (ax2_0_3_30 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_30 * 32) + (ax2_0_3_30 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_68 = 0; ax0_ax1_fused_0_68 < 4; ++ax0_ax1_fused_0_68) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_68 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_68 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1088))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_69 = 0; ax0_ax1_fused_0_69 < 4; ++ax0_ax1_fused_0_69) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_69 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_69 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1088))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_31 = 0; ax3_0_1_31 < 2; ++ax3_0_1_31) {
    for (int ax0_0_62 = 0; ax0_0_62 < 4; ++ax0_0_62) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_62 * 512)) + (ax3_0_1_31 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_62 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_62 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_62 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_62 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_63 = 0; ax0_0_63 < 4; ++ax0_0_63) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_63 * 512)) + (ax3_0_1_31 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_63 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_63 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_63 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_63 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_31 = 0; ax1_0_3_31 < 4; ++ax1_0_3_31) {
      for (int ax2_0_3_31 = 0; ax2_0_3_31 < 4; ++ax2_0_3_31) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_31 * 32) + (ax2_0_3_31 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_31 * 32) + (ax2_0_3_31 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_31 * 32) + (ax2_0_3_31 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_31 * 32) + (ax2_0_3_31 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_31 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_31 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_31 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_31 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_31 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_31 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_31 * 32) + (ax2_0_3_31 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_31 * 32) + (ax2_0_3_31 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_31 * 32) + (ax2_0_3_31 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_31 * 32) + (ax2_0_3_31 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_31 * 32) + (ax2_0_3_31 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_31 * 32) + (ax2_0_3_31 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_31 * 32) + (ax2_0_3_31 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_31 * 32) + (ax2_0_3_31 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_31 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_31 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_31 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_31 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_31 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_31 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_31 * 32) + (ax2_0_3_31 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_31 * 32) + (ax2_0_3_31 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_31 * 32) + (ax2_0_3_31 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_31 * 32) + (ax2_0_3_31 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_70 = 0; ax0_ax1_fused_0_70 < 4; ++ax0_ax1_fused_0_70) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_70 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_70 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1120))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_71 = 0; ax0_ax1_fused_0_71 < 4; ++ax0_ax1_fused_0_71) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_71 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_71 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1120))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_32 = 0; ax3_0_1_32 < 2; ++ax3_0_1_32) {
    for (int ax0_0_64 = 0; ax0_0_64 < 4; ++ax0_0_64) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_64 * 512)) + (ax3_0_1_32 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_64 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_64 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_64 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_64 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_65 = 0; ax0_0_65 < 4; ++ax0_0_65) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_65 * 512)) + (ax3_0_1_32 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_65 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_65 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_65 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_65 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_32 = 0; ax1_0_3_32 < 4; ++ax1_0_3_32) {
      for (int ax2_0_3_32 = 0; ax2_0_3_32 < 4; ++ax2_0_3_32) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_32 * 32) + (ax2_0_3_32 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_32 * 32) + (ax2_0_3_32 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_32 * 32) + (ax2_0_3_32 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_32 * 32) + (ax2_0_3_32 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_32 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_32 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_32 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_32 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_32 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_32 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_32 * 32) + (ax2_0_3_32 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_32 * 32) + (ax2_0_3_32 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_32 * 32) + (ax2_0_3_32 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_32 * 32) + (ax2_0_3_32 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_32 * 32) + (ax2_0_3_32 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_32 * 32) + (ax2_0_3_32 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_32 * 32) + (ax2_0_3_32 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_32 * 32) + (ax2_0_3_32 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_32 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_32 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_32 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_32 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_32 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_32 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_32 * 32) + (ax2_0_3_32 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_32 * 32) + (ax2_0_3_32 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_32 * 32) + (ax2_0_3_32 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_32 * 32) + (ax2_0_3_32 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_72 = 0; ax0_ax1_fused_0_72 < 4; ++ax0_ax1_fused_0_72) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_72 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_72 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1152))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_73 = 0; ax0_ax1_fused_0_73 < 4; ++ax0_ax1_fused_0_73) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_73 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_73 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1152))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_33 = 0; ax3_0_1_33 < 2; ++ax3_0_1_33) {
    for (int ax0_0_66 = 0; ax0_0_66 < 4; ++ax0_0_66) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_66 * 512)) + (ax3_0_1_33 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_66 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_66 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_66 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_66 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_67 = 0; ax0_0_67 < 4; ++ax0_0_67) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_67 * 512)) + (ax3_0_1_33 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_67 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_67 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_67 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_67 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_33 = 0; ax1_0_3_33 < 4; ++ax1_0_3_33) {
      for (int ax2_0_3_33 = 0; ax2_0_3_33 < 4; ++ax2_0_3_33) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_33 * 32) + (ax2_0_3_33 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_33 * 32) + (ax2_0_3_33 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_33 * 32) + (ax2_0_3_33 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_33 * 32) + (ax2_0_3_33 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_33 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_33 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_33 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_33 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_33 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_33 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_33 * 32) + (ax2_0_3_33 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_33 * 32) + (ax2_0_3_33 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_33 * 32) + (ax2_0_3_33 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_33 * 32) + (ax2_0_3_33 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_33 * 32) + (ax2_0_3_33 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_33 * 32) + (ax2_0_3_33 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_33 * 32) + (ax2_0_3_33 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_33 * 32) + (ax2_0_3_33 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_33 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_33 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_33 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_33 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_33 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_33 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_33 * 32) + (ax2_0_3_33 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_33 * 32) + (ax2_0_3_33 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_33 * 32) + (ax2_0_3_33 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_33 * 32) + (ax2_0_3_33 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_74 = 0; ax0_ax1_fused_0_74 < 4; ++ax0_ax1_fused_0_74) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_74 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_74 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1184))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_75 = 0; ax0_ax1_fused_0_75 < 4; ++ax0_ax1_fused_0_75) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_75 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_75 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1184))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_34 = 0; ax3_0_1_34 < 2; ++ax3_0_1_34) {
    for (int ax0_0_68 = 0; ax0_0_68 < 4; ++ax0_0_68) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_68 * 512)) + (ax3_0_1_34 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_68 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_68 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_68 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_68 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_69 = 0; ax0_0_69 < 4; ++ax0_0_69) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_69 * 512)) + (ax3_0_1_34 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_69 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_69 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_69 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_69 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_34 = 0; ax1_0_3_34 < 4; ++ax1_0_3_34) {
      for (int ax2_0_3_34 = 0; ax2_0_3_34 < 4; ++ax2_0_3_34) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_34 * 32) + (ax2_0_3_34 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_34 * 32) + (ax2_0_3_34 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_34 * 32) + (ax2_0_3_34 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_34 * 32) + (ax2_0_3_34 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_34 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_34 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_34 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_34 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_34 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_34 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_34 * 32) + (ax2_0_3_34 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_34 * 32) + (ax2_0_3_34 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_34 * 32) + (ax2_0_3_34 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_34 * 32) + (ax2_0_3_34 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_34 * 32) + (ax2_0_3_34 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_34 * 32) + (ax2_0_3_34 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_34 * 32) + (ax2_0_3_34 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_34 * 32) + (ax2_0_3_34 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_34 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_34 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_34 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_34 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_34 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_34 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_34 * 32) + (ax2_0_3_34 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_34 * 32) + (ax2_0_3_34 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_34 * 32) + (ax2_0_3_34 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_34 * 32) + (ax2_0_3_34 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_76 = 0; ax0_ax1_fused_0_76 < 4; ++ax0_ax1_fused_0_76) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_76 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_76 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1216))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_77 = 0; ax0_ax1_fused_0_77 < 4; ++ax0_ax1_fused_0_77) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_77 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_77 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1216))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_35 = 0; ax3_0_1_35 < 2; ++ax3_0_1_35) {
    for (int ax0_0_70 = 0; ax0_0_70 < 4; ++ax0_0_70) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_70 * 512)) + (ax3_0_1_35 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_70 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_70 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_70 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_70 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_71 = 0; ax0_0_71 < 4; ++ax0_0_71) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_71 * 512)) + (ax3_0_1_35 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_71 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_71 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_71 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_71 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_35 = 0; ax1_0_3_35 < 4; ++ax1_0_3_35) {
      for (int ax2_0_3_35 = 0; ax2_0_3_35 < 4; ++ax2_0_3_35) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_35 * 32) + (ax2_0_3_35 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_35 * 32) + (ax2_0_3_35 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_35 * 32) + (ax2_0_3_35 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_35 * 32) + (ax2_0_3_35 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_35 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_35 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_35 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_35 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_35 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_35 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_35 * 32) + (ax2_0_3_35 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_35 * 32) + (ax2_0_3_35 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_35 * 32) + (ax2_0_3_35 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_35 * 32) + (ax2_0_3_35 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_35 * 32) + (ax2_0_3_35 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_35 * 32) + (ax2_0_3_35 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_35 * 32) + (ax2_0_3_35 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_35 * 32) + (ax2_0_3_35 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_35 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_35 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_35 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_35 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_35 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_35 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_35 * 32) + (ax2_0_3_35 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_35 * 32) + (ax2_0_3_35 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_35 * 32) + (ax2_0_3_35 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_35 * 32) + (ax2_0_3_35 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_78 = 0; ax0_ax1_fused_0_78 < 4; ++ax0_ax1_fused_0_78) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_78 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_78 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1248))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_79 = 0; ax0_ax1_fused_0_79 < 4; ++ax0_ax1_fused_0_79) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_79 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_79 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1248))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_36 = 0; ax3_0_1_36 < 2; ++ax3_0_1_36) {
    for (int ax0_0_72 = 0; ax0_0_72 < 4; ++ax0_0_72) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_72 * 512)) + (ax3_0_1_36 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_72 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_72 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_72 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_72 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_73 = 0; ax0_0_73 < 4; ++ax0_0_73) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_73 * 512)) + (ax3_0_1_36 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_73 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_73 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_73 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_73 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_36 = 0; ax1_0_3_36 < 4; ++ax1_0_3_36) {
      for (int ax2_0_3_36 = 0; ax2_0_3_36 < 4; ++ax2_0_3_36) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_36 * 32) + (ax2_0_3_36 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_36 * 32) + (ax2_0_3_36 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_36 * 32) + (ax2_0_3_36 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_36 * 32) + (ax2_0_3_36 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_36 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_36 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_36 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_36 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_36 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_36 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_36 * 32) + (ax2_0_3_36 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_36 * 32) + (ax2_0_3_36 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_36 * 32) + (ax2_0_3_36 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_36 * 32) + (ax2_0_3_36 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_36 * 32) + (ax2_0_3_36 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_36 * 32) + (ax2_0_3_36 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_36 * 32) + (ax2_0_3_36 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_36 * 32) + (ax2_0_3_36 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_36 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_36 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_36 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_36 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_36 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_36 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_36 * 32) + (ax2_0_3_36 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_36 * 32) + (ax2_0_3_36 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_36 * 32) + (ax2_0_3_36 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_36 * 32) + (ax2_0_3_36 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_80 = 0; ax0_ax1_fused_0_80 < 4; ++ax0_ax1_fused_0_80) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_80 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_80 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1280))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_81 = 0; ax0_ax1_fused_0_81 < 4; ++ax0_ax1_fused_0_81) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_81 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_81 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1280))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_37 = 0; ax3_0_1_37 < 2; ++ax3_0_1_37) {
    for (int ax0_0_74 = 0; ax0_0_74 < 4; ++ax0_0_74) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_74 * 512)) + (ax3_0_1_37 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_74 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_74 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_74 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_74 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_75 = 0; ax0_0_75 < 4; ++ax0_0_75) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_75 * 512)) + (ax3_0_1_37 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_75 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_75 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_75 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_75 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_37 = 0; ax1_0_3_37 < 4; ++ax1_0_3_37) {
      for (int ax2_0_3_37 = 0; ax2_0_3_37 < 4; ++ax2_0_3_37) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_37 * 32) + (ax2_0_3_37 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_37 * 32) + (ax2_0_3_37 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_37 * 32) + (ax2_0_3_37 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_37 * 32) + (ax2_0_3_37 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_37 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_37 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_37 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_37 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_37 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_37 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_37 * 32) + (ax2_0_3_37 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_37 * 32) + (ax2_0_3_37 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_37 * 32) + (ax2_0_3_37 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_37 * 32) + (ax2_0_3_37 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_37 * 32) + (ax2_0_3_37 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_37 * 32) + (ax2_0_3_37 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_37 * 32) + (ax2_0_3_37 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_37 * 32) + (ax2_0_3_37 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_37 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_37 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_37 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_37 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_37 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_37 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_37 * 32) + (ax2_0_3_37 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_37 * 32) + (ax2_0_3_37 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_37 * 32) + (ax2_0_3_37 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_37 * 32) + (ax2_0_3_37 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_82 = 0; ax0_ax1_fused_0_82 < 4; ++ax0_ax1_fused_0_82) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_82 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_82 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1312))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_83 = 0; ax0_ax1_fused_0_83 < 4; ++ax0_ax1_fused_0_83) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_83 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_83 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1312))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_38 = 0; ax3_0_1_38 < 2; ++ax3_0_1_38) {
    for (int ax0_0_76 = 0; ax0_0_76 < 4; ++ax0_0_76) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_76 * 512)) + (ax3_0_1_38 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_76 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_76 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_76 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_76 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_77 = 0; ax0_0_77 < 4; ++ax0_0_77) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_77 * 512)) + (ax3_0_1_38 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_77 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_77 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_77 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_77 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_38 = 0; ax1_0_3_38 < 4; ++ax1_0_3_38) {
      for (int ax2_0_3_38 = 0; ax2_0_3_38 < 4; ++ax2_0_3_38) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_38 * 32) + (ax2_0_3_38 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_38 * 32) + (ax2_0_3_38 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_38 * 32) + (ax2_0_3_38 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_38 * 32) + (ax2_0_3_38 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_38 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_38 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_38 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_38 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_38 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_38 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_38 * 32) + (ax2_0_3_38 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_38 * 32) + (ax2_0_3_38 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_38 * 32) + (ax2_0_3_38 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_38 * 32) + (ax2_0_3_38 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_38 * 32) + (ax2_0_3_38 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_38 * 32) + (ax2_0_3_38 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_38 * 32) + (ax2_0_3_38 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_38 * 32) + (ax2_0_3_38 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_38 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_38 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_38 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_38 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_38 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_38 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_38 * 32) + (ax2_0_3_38 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_38 * 32) + (ax2_0_3_38 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_38 * 32) + (ax2_0_3_38 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_38 * 32) + (ax2_0_3_38 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_84 = 0; ax0_ax1_fused_0_84 < 4; ++ax0_ax1_fused_0_84) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_84 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_84 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1344))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_85 = 0; ax0_ax1_fused_0_85 < 4; ++ax0_ax1_fused_0_85) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_85 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_85 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1344))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_39 = 0; ax3_0_1_39 < 2; ++ax3_0_1_39) {
    for (int ax0_0_78 = 0; ax0_0_78 < 4; ++ax0_0_78) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_78 * 512)) + (ax3_0_1_39 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_78 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_78 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_78 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_78 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_79 = 0; ax0_0_79 < 4; ++ax0_0_79) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_79 * 512)) + (ax3_0_1_39 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_79 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_79 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_79 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_79 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_39 = 0; ax1_0_3_39 < 4; ++ax1_0_3_39) {
      for (int ax2_0_3_39 = 0; ax2_0_3_39 < 4; ++ax2_0_3_39) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_39 * 32) + (ax2_0_3_39 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_39 * 32) + (ax2_0_3_39 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_39 * 32) + (ax2_0_3_39 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_39 * 32) + (ax2_0_3_39 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_39 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_39 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_39 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_39 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_39 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_39 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_39 * 32) + (ax2_0_3_39 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_39 * 32) + (ax2_0_3_39 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_39 * 32) + (ax2_0_3_39 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_39 * 32) + (ax2_0_3_39 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_39 * 32) + (ax2_0_3_39 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_39 * 32) + (ax2_0_3_39 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_39 * 32) + (ax2_0_3_39 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_39 * 32) + (ax2_0_3_39 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_39 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_39 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_39 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_39 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_39 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_39 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_39 * 32) + (ax2_0_3_39 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_39 * 32) + (ax2_0_3_39 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_39 * 32) + (ax2_0_3_39 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_39 * 32) + (ax2_0_3_39 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_86 = 0; ax0_ax1_fused_0_86 < 4; ++ax0_ax1_fused_0_86) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_86 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_86 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1376))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_87 = 0; ax0_ax1_fused_0_87 < 4; ++ax0_ax1_fused_0_87) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_87 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_87 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1376))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_40 = 0; ax3_0_1_40 < 2; ++ax3_0_1_40) {
    for (int ax0_0_80 = 0; ax0_0_80 < 4; ++ax0_0_80) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_80 * 512)) + (ax3_0_1_40 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_80 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_80 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_80 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_80 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_81 = 0; ax0_0_81 < 4; ++ax0_0_81) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_81 * 512)) + (ax3_0_1_40 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_81 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_81 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_81 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_81 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_40 = 0; ax1_0_3_40 < 4; ++ax1_0_3_40) {
      for (int ax2_0_3_40 = 0; ax2_0_3_40 < 4; ++ax2_0_3_40) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_40 * 32) + (ax2_0_3_40 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_40 * 32) + (ax2_0_3_40 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_40 * 32) + (ax2_0_3_40 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_40 * 32) + (ax2_0_3_40 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_40 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_40 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_40 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_40 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_40 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_40 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_40 * 32) + (ax2_0_3_40 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_40 * 32) + (ax2_0_3_40 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_40 * 32) + (ax2_0_3_40 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_40 * 32) + (ax2_0_3_40 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_40 * 32) + (ax2_0_3_40 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_40 * 32) + (ax2_0_3_40 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_40 * 32) + (ax2_0_3_40 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_40 * 32) + (ax2_0_3_40 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_40 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_40 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_40 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_40 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_40 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_40 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_40 * 32) + (ax2_0_3_40 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_40 * 32) + (ax2_0_3_40 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_40 * 32) + (ax2_0_3_40 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_40 * 32) + (ax2_0_3_40 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_88 = 0; ax0_ax1_fused_0_88 < 4; ++ax0_ax1_fused_0_88) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_88 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_88 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1408))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_89 = 0; ax0_ax1_fused_0_89 < 4; ++ax0_ax1_fused_0_89) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_89 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_89 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1408))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_41 = 0; ax3_0_1_41 < 2; ++ax3_0_1_41) {
    for (int ax0_0_82 = 0; ax0_0_82 < 4; ++ax0_0_82) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_82 * 512)) + (ax3_0_1_41 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_82 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_82 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_82 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_82 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_83 = 0; ax0_0_83 < 4; ++ax0_0_83) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_83 * 512)) + (ax3_0_1_41 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_83 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_83 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_83 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_83 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_41 = 0; ax1_0_3_41 < 4; ++ax1_0_3_41) {
      for (int ax2_0_3_41 = 0; ax2_0_3_41 < 4; ++ax2_0_3_41) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_41 * 32) + (ax2_0_3_41 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_41 * 32) + (ax2_0_3_41 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_41 * 32) + (ax2_0_3_41 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_41 * 32) + (ax2_0_3_41 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_41 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_41 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_41 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_41 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_41 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_41 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_41 * 32) + (ax2_0_3_41 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_41 * 32) + (ax2_0_3_41 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_41 * 32) + (ax2_0_3_41 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_41 * 32) + (ax2_0_3_41 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_41 * 32) + (ax2_0_3_41 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_41 * 32) + (ax2_0_3_41 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_41 * 32) + (ax2_0_3_41 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_41 * 32) + (ax2_0_3_41 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_41 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_41 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_41 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_41 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_41 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_41 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_41 * 32) + (ax2_0_3_41 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_41 * 32) + (ax2_0_3_41 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_41 * 32) + (ax2_0_3_41 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_41 * 32) + (ax2_0_3_41 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_90 = 0; ax0_ax1_fused_0_90 < 4; ++ax0_ax1_fused_0_90) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_90 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_90 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1440))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_91 = 0; ax0_ax1_fused_0_91 < 4; ++ax0_ax1_fused_0_91) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_91 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_91 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1440))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_42 = 0; ax3_0_1_42 < 2; ++ax3_0_1_42) {
    for (int ax0_0_84 = 0; ax0_0_84 < 4; ++ax0_0_84) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_84 * 512)) + (ax3_0_1_42 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_84 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_84 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_84 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_84 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_85 = 0; ax0_0_85 < 4; ++ax0_0_85) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_85 * 512)) + (ax3_0_1_42 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_85 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_85 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_85 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_85 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_42 = 0; ax1_0_3_42 < 4; ++ax1_0_3_42) {
      for (int ax2_0_3_42 = 0; ax2_0_3_42 < 4; ++ax2_0_3_42) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_42 * 32) + (ax2_0_3_42 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_42 * 32) + (ax2_0_3_42 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_42 * 32) + (ax2_0_3_42 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_42 * 32) + (ax2_0_3_42 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_42 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_42 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_42 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_42 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_42 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_42 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_42 * 32) + (ax2_0_3_42 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_42 * 32) + (ax2_0_3_42 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_42 * 32) + (ax2_0_3_42 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_42 * 32) + (ax2_0_3_42 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_42 * 32) + (ax2_0_3_42 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_42 * 32) + (ax2_0_3_42 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_42 * 32) + (ax2_0_3_42 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_42 * 32) + (ax2_0_3_42 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_42 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_42 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_42 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_42 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_42 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_42 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_42 * 32) + (ax2_0_3_42 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_42 * 32) + (ax2_0_3_42 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_42 * 32) + (ax2_0_3_42 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_42 * 32) + (ax2_0_3_42 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_92 = 0; ax0_ax1_fused_0_92 < 4; ++ax0_ax1_fused_0_92) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_92 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_92 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1472))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_93 = 0; ax0_ax1_fused_0_93 < 4; ++ax0_ax1_fused_0_93) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_93 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_93 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1472))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_43 = 0; ax3_0_1_43 < 2; ++ax3_0_1_43) {
    for (int ax0_0_86 = 0; ax0_0_86 < 4; ++ax0_0_86) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_86 * 512)) + (ax3_0_1_43 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_86 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_86 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_86 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_86 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_87 = 0; ax0_0_87 < 4; ++ax0_0_87) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_87 * 512)) + (ax3_0_1_43 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_87 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_87 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_87 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_87 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_43 = 0; ax1_0_3_43 < 4; ++ax1_0_3_43) {
      for (int ax2_0_3_43 = 0; ax2_0_3_43 < 4; ++ax2_0_3_43) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_43 * 32) + (ax2_0_3_43 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_43 * 32) + (ax2_0_3_43 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_43 * 32) + (ax2_0_3_43 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_43 * 32) + (ax2_0_3_43 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_43 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_43 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_43 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_43 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_43 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_43 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_43 * 32) + (ax2_0_3_43 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_43 * 32) + (ax2_0_3_43 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_43 * 32) + (ax2_0_3_43 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_43 * 32) + (ax2_0_3_43 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_43 * 32) + (ax2_0_3_43 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_43 * 32) + (ax2_0_3_43 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_43 * 32) + (ax2_0_3_43 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_43 * 32) + (ax2_0_3_43 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_43 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_43 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_43 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_43 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_43 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_43 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_43 * 32) + (ax2_0_3_43 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_43 * 32) + (ax2_0_3_43 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_43 * 32) + (ax2_0_3_43 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_43 * 32) + (ax2_0_3_43 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_94 = 0; ax0_ax1_fused_0_94 < 4; ++ax0_ax1_fused_0_94) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_94 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_94 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1504))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_95 = 0; ax0_ax1_fused_0_95 < 4; ++ax0_ax1_fused_0_95) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_95 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_95 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1504))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_44 = 0; ax3_0_1_44 < 2; ++ax3_0_1_44) {
    for (int ax0_0_88 = 0; ax0_0_88 < 4; ++ax0_0_88) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_88 * 512)) + (ax3_0_1_44 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_88 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_88 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_88 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_88 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_89 = 0; ax0_0_89 < 4; ++ax0_0_89) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_89 * 512)) + (ax3_0_1_44 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_89 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_89 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_89 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_89 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_44 = 0; ax1_0_3_44 < 4; ++ax1_0_3_44) {
      for (int ax2_0_3_44 = 0; ax2_0_3_44 < 4; ++ax2_0_3_44) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_44 * 32) + (ax2_0_3_44 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_44 * 32) + (ax2_0_3_44 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_44 * 32) + (ax2_0_3_44 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_44 * 32) + (ax2_0_3_44 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_44 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_44 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_44 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_44 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_44 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_44 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_44 * 32) + (ax2_0_3_44 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_44 * 32) + (ax2_0_3_44 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_44 * 32) + (ax2_0_3_44 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_44 * 32) + (ax2_0_3_44 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_44 * 32) + (ax2_0_3_44 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_44 * 32) + (ax2_0_3_44 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_44 * 32) + (ax2_0_3_44 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_44 * 32) + (ax2_0_3_44 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_44 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_44 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_44 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_44 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_44 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_44 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_44 * 32) + (ax2_0_3_44 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_44 * 32) + (ax2_0_3_44 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_44 * 32) + (ax2_0_3_44 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_44 * 32) + (ax2_0_3_44 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_96 = 0; ax0_ax1_fused_0_96 < 4; ++ax0_ax1_fused_0_96) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_96 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_96 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1536))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_97 = 0; ax0_ax1_fused_0_97 < 4; ++ax0_ax1_fused_0_97) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_97 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_97 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1536))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_45 = 0; ax3_0_1_45 < 2; ++ax3_0_1_45) {
    for (int ax0_0_90 = 0; ax0_0_90 < 4; ++ax0_0_90) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_90 * 512)) + (ax3_0_1_45 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_90 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_90 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_90 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_90 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_91 = 0; ax0_0_91 < 4; ++ax0_0_91) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_91 * 512)) + (ax3_0_1_45 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_91 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_91 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_91 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_91 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_45 = 0; ax1_0_3_45 < 4; ++ax1_0_3_45) {
      for (int ax2_0_3_45 = 0; ax2_0_3_45 < 4; ++ax2_0_3_45) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_45 * 32) + (ax2_0_3_45 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_45 * 32) + (ax2_0_3_45 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_45 * 32) + (ax2_0_3_45 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_45 * 32) + (ax2_0_3_45 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_45 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_45 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_45 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_45 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_45 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_45 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_45 * 32) + (ax2_0_3_45 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_45 * 32) + (ax2_0_3_45 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_45 * 32) + (ax2_0_3_45 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_45 * 32) + (ax2_0_3_45 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_45 * 32) + (ax2_0_3_45 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_45 * 32) + (ax2_0_3_45 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_45 * 32) + (ax2_0_3_45 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_45 * 32) + (ax2_0_3_45 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_45 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_45 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_45 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_45 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_45 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_45 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_45 * 32) + (ax2_0_3_45 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_45 * 32) + (ax2_0_3_45 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_45 * 32) + (ax2_0_3_45 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_45 * 32) + (ax2_0_3_45 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_98 = 0; ax0_ax1_fused_0_98 < 4; ++ax0_ax1_fused_0_98) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_98 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_98 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1568))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_99 = 0; ax0_ax1_fused_0_99 < 4; ++ax0_ax1_fused_0_99) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_99 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_99 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1568))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_46 = 0; ax3_0_1_46 < 2; ++ax3_0_1_46) {
    for (int ax0_0_92 = 0; ax0_0_92 < 4; ++ax0_0_92) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_92 * 512)) + (ax3_0_1_46 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_92 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_92 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_92 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_92 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_93 = 0; ax0_0_93 < 4; ++ax0_0_93) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_93 * 512)) + (ax3_0_1_46 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_93 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_93 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_93 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_93 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_46 = 0; ax1_0_3_46 < 4; ++ax1_0_3_46) {
      for (int ax2_0_3_46 = 0; ax2_0_3_46 < 4; ++ax2_0_3_46) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_46 * 32) + (ax2_0_3_46 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_46 * 32) + (ax2_0_3_46 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_46 * 32) + (ax2_0_3_46 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_46 * 32) + (ax2_0_3_46 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_46 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_46 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_46 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_46 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_46 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_46 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_46 * 32) + (ax2_0_3_46 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_46 * 32) + (ax2_0_3_46 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_46 * 32) + (ax2_0_3_46 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_46 * 32) + (ax2_0_3_46 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_46 * 32) + (ax2_0_3_46 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_46 * 32) + (ax2_0_3_46 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_46 * 32) + (ax2_0_3_46 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_46 * 32) + (ax2_0_3_46 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_46 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_46 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_46 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_46 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_46 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_46 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_46 * 32) + (ax2_0_3_46 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_46 * 32) + (ax2_0_3_46 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_46 * 32) + (ax2_0_3_46 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_46 * 32) + (ax2_0_3_46 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_100 = 0; ax0_ax1_fused_0_100 < 4; ++ax0_ax1_fused_0_100) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_100 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_100 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1600))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_101 = 0; ax0_ax1_fused_0_101 < 4; ++ax0_ax1_fused_0_101) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_101 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_101 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1600))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_47 = 0; ax3_0_1_47 < 2; ++ax3_0_1_47) {
    for (int ax0_0_94 = 0; ax0_0_94 < 4; ++ax0_0_94) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_94 * 512)) + (ax3_0_1_47 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_94 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_94 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_94 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_94 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_95 = 0; ax0_0_95 < 4; ++ax0_0_95) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_95 * 512)) + (ax3_0_1_47 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_95 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_95 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_95 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_95 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_47 = 0; ax1_0_3_47 < 4; ++ax1_0_3_47) {
      for (int ax2_0_3_47 = 0; ax2_0_3_47 < 4; ++ax2_0_3_47) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_47 * 32) + (ax2_0_3_47 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_47 * 32) + (ax2_0_3_47 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_47 * 32) + (ax2_0_3_47 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_47 * 32) + (ax2_0_3_47 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_47 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_47 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_47 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_47 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_47 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_47 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_47 * 32) + (ax2_0_3_47 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_47 * 32) + (ax2_0_3_47 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_47 * 32) + (ax2_0_3_47 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_47 * 32) + (ax2_0_3_47 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_47 * 32) + (ax2_0_3_47 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_47 * 32) + (ax2_0_3_47 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_47 * 32) + (ax2_0_3_47 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_47 * 32) + (ax2_0_3_47 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_47 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_47 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_47 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_47 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_47 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_47 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_47 * 32) + (ax2_0_3_47 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_47 * 32) + (ax2_0_3_47 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_47 * 32) + (ax2_0_3_47 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_47 * 32) + (ax2_0_3_47 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_102 = 0; ax0_ax1_fused_0_102 < 4; ++ax0_ax1_fused_0_102) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_102 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_102 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1632))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_103 = 0; ax0_ax1_fused_0_103 < 4; ++ax0_ax1_fused_0_103) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_103 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_103 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1632))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_48 = 0; ax3_0_1_48 < 2; ++ax3_0_1_48) {
    for (int ax0_0_96 = 0; ax0_0_96 < 4; ++ax0_0_96) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_96 * 512)) + (ax3_0_1_48 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_96 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_96 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_96 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_96 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_97 = 0; ax0_0_97 < 4; ++ax0_0_97) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_97 * 512)) + (ax3_0_1_48 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_97 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_97 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_97 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_97 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_48 = 0; ax1_0_3_48 < 4; ++ax1_0_3_48) {
      for (int ax2_0_3_48 = 0; ax2_0_3_48 < 4; ++ax2_0_3_48) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_48 * 32) + (ax2_0_3_48 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_48 * 32) + (ax2_0_3_48 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_48 * 32) + (ax2_0_3_48 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_48 * 32) + (ax2_0_3_48 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_48 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_48 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_48 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_48 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_48 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_48 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_48 * 32) + (ax2_0_3_48 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_48 * 32) + (ax2_0_3_48 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_48 * 32) + (ax2_0_3_48 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_48 * 32) + (ax2_0_3_48 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_48 * 32) + (ax2_0_3_48 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_48 * 32) + (ax2_0_3_48 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_48 * 32) + (ax2_0_3_48 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_48 * 32) + (ax2_0_3_48 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_48 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_48 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_48 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_48 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_48 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_48 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_48 * 32) + (ax2_0_3_48 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_48 * 32) + (ax2_0_3_48 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_48 * 32) + (ax2_0_3_48 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_48 * 32) + (ax2_0_3_48 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_104 = 0; ax0_ax1_fused_0_104 < 4; ++ax0_ax1_fused_0_104) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_104 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_104 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1664))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_105 = 0; ax0_ax1_fused_0_105 < 4; ++ax0_ax1_fused_0_105) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_105 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_105 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1664))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_49 = 0; ax3_0_1_49 < 2; ++ax3_0_1_49) {
    for (int ax0_0_98 = 0; ax0_0_98 < 4; ++ax0_0_98) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_98 * 512)) + (ax3_0_1_49 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_98 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_98 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_98 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_98 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_99 = 0; ax0_0_99 < 4; ++ax0_0_99) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_99 * 512)) + (ax3_0_1_49 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_99 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_99 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_99 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_99 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_49 = 0; ax1_0_3_49 < 4; ++ax1_0_3_49) {
      for (int ax2_0_3_49 = 0; ax2_0_3_49 < 4; ++ax2_0_3_49) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_49 * 32) + (ax2_0_3_49 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_49 * 32) + (ax2_0_3_49 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_49 * 32) + (ax2_0_3_49 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_49 * 32) + (ax2_0_3_49 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_49 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_49 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_49 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_49 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_49 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_49 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_49 * 32) + (ax2_0_3_49 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_49 * 32) + (ax2_0_3_49 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_49 * 32) + (ax2_0_3_49 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_49 * 32) + (ax2_0_3_49 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_49 * 32) + (ax2_0_3_49 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_49 * 32) + (ax2_0_3_49 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_49 * 32) + (ax2_0_3_49 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_49 * 32) + (ax2_0_3_49 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_49 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_49 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_49 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_49 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_49 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_49 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_49 * 32) + (ax2_0_3_49 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_49 * 32) + (ax2_0_3_49 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_49 * 32) + (ax2_0_3_49 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_49 * 32) + (ax2_0_3_49 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_106 = 0; ax0_ax1_fused_0_106 < 4; ++ax0_ax1_fused_0_106) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_106 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_106 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1696))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_107 = 0; ax0_ax1_fused_0_107 < 4; ++ax0_ax1_fused_0_107) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_107 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_107 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1696))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_50 = 0; ax3_0_1_50 < 2; ++ax3_0_1_50) {
    for (int ax0_0_100 = 0; ax0_0_100 < 4; ++ax0_0_100) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_100 * 512)) + (ax3_0_1_50 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_100 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_100 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_100 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_100 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_101 = 0; ax0_0_101 < 4; ++ax0_0_101) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_101 * 512)) + (ax3_0_1_50 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_101 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_101 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_101 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_101 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_50 = 0; ax1_0_3_50 < 4; ++ax1_0_3_50) {
      for (int ax2_0_3_50 = 0; ax2_0_3_50 < 4; ++ax2_0_3_50) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_50 * 32) + (ax2_0_3_50 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_50 * 32) + (ax2_0_3_50 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_50 * 32) + (ax2_0_3_50 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_50 * 32) + (ax2_0_3_50 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_50 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_50 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_50 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_50 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_50 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_50 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_50 * 32) + (ax2_0_3_50 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_50 * 32) + (ax2_0_3_50 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_50 * 32) + (ax2_0_3_50 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_50 * 32) + (ax2_0_3_50 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_50 * 32) + (ax2_0_3_50 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_50 * 32) + (ax2_0_3_50 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_50 * 32) + (ax2_0_3_50 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_50 * 32) + (ax2_0_3_50 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_50 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_50 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_50 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_50 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_50 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_50 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_50 * 32) + (ax2_0_3_50 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_50 * 32) + (ax2_0_3_50 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_50 * 32) + (ax2_0_3_50 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_50 * 32) + (ax2_0_3_50 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_108 = 0; ax0_ax1_fused_0_108 < 4; ++ax0_ax1_fused_0_108) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_108 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_108 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1728))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_109 = 0; ax0_ax1_fused_0_109 < 4; ++ax0_ax1_fused_0_109) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_109 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_109 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1728))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_51 = 0; ax3_0_1_51 < 2; ++ax3_0_1_51) {
    for (int ax0_0_102 = 0; ax0_0_102 < 4; ++ax0_0_102) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_102 * 512)) + (ax3_0_1_51 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_102 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_102 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_102 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_102 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_103 = 0; ax0_0_103 < 4; ++ax0_0_103) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_103 * 512)) + (ax3_0_1_51 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_103 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_103 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_103 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_103 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_51 = 0; ax1_0_3_51 < 4; ++ax1_0_3_51) {
      for (int ax2_0_3_51 = 0; ax2_0_3_51 < 4; ++ax2_0_3_51) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_51 * 32) + (ax2_0_3_51 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_51 * 32) + (ax2_0_3_51 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_51 * 32) + (ax2_0_3_51 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_51 * 32) + (ax2_0_3_51 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_51 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_51 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_51 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_51 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_51 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_51 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_51 * 32) + (ax2_0_3_51 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_51 * 32) + (ax2_0_3_51 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_51 * 32) + (ax2_0_3_51 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_51 * 32) + (ax2_0_3_51 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_51 * 32) + (ax2_0_3_51 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_51 * 32) + (ax2_0_3_51 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_51 * 32) + (ax2_0_3_51 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_51 * 32) + (ax2_0_3_51 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_51 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_51 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_51 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_51 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_51 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_51 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_51 * 32) + (ax2_0_3_51 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_51 * 32) + (ax2_0_3_51 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_51 * 32) + (ax2_0_3_51 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_51 * 32) + (ax2_0_3_51 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_110 = 0; ax0_ax1_fused_0_110 < 4; ++ax0_ax1_fused_0_110) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_110 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_110 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1760))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_111 = 0; ax0_ax1_fused_0_111 < 4; ++ax0_ax1_fused_0_111) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_111 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_111 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1760))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_52 = 0; ax3_0_1_52 < 2; ++ax3_0_1_52) {
    for (int ax0_0_104 = 0; ax0_0_104 < 4; ++ax0_0_104) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_104 * 512)) + (ax3_0_1_52 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_104 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_104 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_104 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_104 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_105 = 0; ax0_0_105 < 4; ++ax0_0_105) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_105 * 512)) + (ax3_0_1_52 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_105 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_105 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_105 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_105 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_52 = 0; ax1_0_3_52 < 4; ++ax1_0_3_52) {
      for (int ax2_0_3_52 = 0; ax2_0_3_52 < 4; ++ax2_0_3_52) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_52 * 32) + (ax2_0_3_52 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_52 * 32) + (ax2_0_3_52 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_52 * 32) + (ax2_0_3_52 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_52 * 32) + (ax2_0_3_52 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_52 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_52 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_52 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_52 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_52 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_52 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_52 * 32) + (ax2_0_3_52 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_52 * 32) + (ax2_0_3_52 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_52 * 32) + (ax2_0_3_52 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_52 * 32) + (ax2_0_3_52 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_52 * 32) + (ax2_0_3_52 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_52 * 32) + (ax2_0_3_52 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_52 * 32) + (ax2_0_3_52 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_52 * 32) + (ax2_0_3_52 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_52 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_52 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_52 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_52 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_52 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_52 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_52 * 32) + (ax2_0_3_52 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_52 * 32) + (ax2_0_3_52 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_52 * 32) + (ax2_0_3_52 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_52 * 32) + (ax2_0_3_52 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_112 = 0; ax0_ax1_fused_0_112 < 4; ++ax0_ax1_fused_0_112) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_112 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_112 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1792))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_113 = 0; ax0_ax1_fused_0_113 < 4; ++ax0_ax1_fused_0_113) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_113 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_113 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1792))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_53 = 0; ax3_0_1_53 < 2; ++ax3_0_1_53) {
    for (int ax0_0_106 = 0; ax0_0_106 < 4; ++ax0_0_106) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_106 * 512)) + (ax3_0_1_53 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_106 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_106 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_106 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_106 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_107 = 0; ax0_0_107 < 4; ++ax0_0_107) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_107 * 512)) + (ax3_0_1_53 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_107 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_107 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_107 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_107 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_53 = 0; ax1_0_3_53 < 4; ++ax1_0_3_53) {
      for (int ax2_0_3_53 = 0; ax2_0_3_53 < 4; ++ax2_0_3_53) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_53 * 32) + (ax2_0_3_53 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_53 * 32) + (ax2_0_3_53 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_53 * 32) + (ax2_0_3_53 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_53 * 32) + (ax2_0_3_53 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_53 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_53 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_53 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_53 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_53 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_53 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_53 * 32) + (ax2_0_3_53 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_53 * 32) + (ax2_0_3_53 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_53 * 32) + (ax2_0_3_53 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_53 * 32) + (ax2_0_3_53 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_53 * 32) + (ax2_0_3_53 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_53 * 32) + (ax2_0_3_53 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_53 * 32) + (ax2_0_3_53 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_53 * 32) + (ax2_0_3_53 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_53 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_53 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_53 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_53 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_53 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_53 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_53 * 32) + (ax2_0_3_53 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_53 * 32) + (ax2_0_3_53 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_53 * 32) + (ax2_0_3_53 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_53 * 32) + (ax2_0_3_53 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_114 = 0; ax0_ax1_fused_0_114 < 4; ++ax0_ax1_fused_0_114) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_114 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_114 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1824))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_115 = 0; ax0_ax1_fused_0_115 < 4; ++ax0_ax1_fused_0_115) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_115 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_115 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1824))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_54 = 0; ax3_0_1_54 < 2; ++ax3_0_1_54) {
    for (int ax0_0_108 = 0; ax0_0_108 < 4; ++ax0_0_108) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_108 * 512)) + (ax3_0_1_54 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_108 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_108 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_108 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_108 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_109 = 0; ax0_0_109 < 4; ++ax0_0_109) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_109 * 512)) + (ax3_0_1_54 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_109 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_109 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_109 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_109 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_54 = 0; ax1_0_3_54 < 4; ++ax1_0_3_54) {
      for (int ax2_0_3_54 = 0; ax2_0_3_54 < 4; ++ax2_0_3_54) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_54 * 32) + (ax2_0_3_54 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_54 * 32) + (ax2_0_3_54 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_54 * 32) + (ax2_0_3_54 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_54 * 32) + (ax2_0_3_54 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_54 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_54 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_54 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_54 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_54 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_54 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_54 * 32) + (ax2_0_3_54 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_54 * 32) + (ax2_0_3_54 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_54 * 32) + (ax2_0_3_54 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_54 * 32) + (ax2_0_3_54 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_54 * 32) + (ax2_0_3_54 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_54 * 32) + (ax2_0_3_54 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_54 * 32) + (ax2_0_3_54 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_54 * 32) + (ax2_0_3_54 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_54 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_54 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_54 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_54 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_54 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_54 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_54 * 32) + (ax2_0_3_54 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_54 * 32) + (ax2_0_3_54 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_54 * 32) + (ax2_0_3_54 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_54 * 32) + (ax2_0_3_54 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_116 = 0; ax0_ax1_fused_0_116 < 4; ++ax0_ax1_fused_0_116) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_116 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_116 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1856))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_117 = 0; ax0_ax1_fused_0_117 < 4; ++ax0_ax1_fused_0_117) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_117 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_117 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1856))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_55 = 0; ax3_0_1_55 < 2; ++ax3_0_1_55) {
    for (int ax0_0_110 = 0; ax0_0_110 < 4; ++ax0_0_110) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_110 * 512)) + (ax3_0_1_55 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_110 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_110 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_110 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_110 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_111 = 0; ax0_0_111 < 4; ++ax0_0_111) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_111 * 512)) + (ax3_0_1_55 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_111 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_111 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_111 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_111 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_55 = 0; ax1_0_3_55 < 4; ++ax1_0_3_55) {
      for (int ax2_0_3_55 = 0; ax2_0_3_55 < 4; ++ax2_0_3_55) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_55 * 32) + (ax2_0_3_55 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_55 * 32) + (ax2_0_3_55 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_55 * 32) + (ax2_0_3_55 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_55 * 32) + (ax2_0_3_55 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_55 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_55 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_55 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_55 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_55 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_55 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_55 * 32) + (ax2_0_3_55 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_55 * 32) + (ax2_0_3_55 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_55 * 32) + (ax2_0_3_55 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_55 * 32) + (ax2_0_3_55 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_55 * 32) + (ax2_0_3_55 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_55 * 32) + (ax2_0_3_55 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_55 * 32) + (ax2_0_3_55 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_55 * 32) + (ax2_0_3_55 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_55 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_55 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_55 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_55 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_55 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_55 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_55 * 32) + (ax2_0_3_55 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_55 * 32) + (ax2_0_3_55 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_55 * 32) + (ax2_0_3_55 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_55 * 32) + (ax2_0_3_55 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_118 = 0; ax0_ax1_fused_0_118 < 4; ++ax0_ax1_fused_0_118) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_118 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_118 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1888))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_119 = 0; ax0_ax1_fused_0_119 < 4; ++ax0_ax1_fused_0_119) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_119 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_119 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1888))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_56 = 0; ax3_0_1_56 < 2; ++ax3_0_1_56) {
    for (int ax0_0_112 = 0; ax0_0_112 < 4; ++ax0_0_112) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_112 * 512)) + (ax3_0_1_56 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_112 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_112 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_112 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_112 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_113 = 0; ax0_0_113 < 4; ++ax0_0_113) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_113 * 512)) + (ax3_0_1_56 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_113 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_113 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_113 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_113 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_56 = 0; ax1_0_3_56 < 4; ++ax1_0_3_56) {
      for (int ax2_0_3_56 = 0; ax2_0_3_56 < 4; ++ax2_0_3_56) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_56 * 32) + (ax2_0_3_56 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_56 * 32) + (ax2_0_3_56 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_56 * 32) + (ax2_0_3_56 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_56 * 32) + (ax2_0_3_56 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_56 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_56 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_56 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_56 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_56 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_56 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_56 * 32) + (ax2_0_3_56 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_56 * 32) + (ax2_0_3_56 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_56 * 32) + (ax2_0_3_56 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_56 * 32) + (ax2_0_3_56 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_56 * 32) + (ax2_0_3_56 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_56 * 32) + (ax2_0_3_56 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_56 * 32) + (ax2_0_3_56 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_56 * 32) + (ax2_0_3_56 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_56 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_56 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_56 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_56 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_56 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_56 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_56 * 32) + (ax2_0_3_56 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_56 * 32) + (ax2_0_3_56 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_56 * 32) + (ax2_0_3_56 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_56 * 32) + (ax2_0_3_56 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_120 = 0; ax0_ax1_fused_0_120 < 4; ++ax0_ax1_fused_0_120) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_120 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_120 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1920))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_121 = 0; ax0_ax1_fused_0_121 < 4; ++ax0_ax1_fused_0_121) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_121 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_121 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1920))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_57 = 0; ax3_0_1_57 < 2; ++ax3_0_1_57) {
    for (int ax0_0_114 = 0; ax0_0_114 < 4; ++ax0_0_114) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_114 * 512)) + (ax3_0_1_57 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_114 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_114 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_114 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_114 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_115 = 0; ax0_0_115 < 4; ++ax0_0_115) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_115 * 512)) + (ax3_0_1_57 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_115 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_115 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_115 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_115 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_57 = 0; ax1_0_3_57 < 4; ++ax1_0_3_57) {
      for (int ax2_0_3_57 = 0; ax2_0_3_57 < 4; ++ax2_0_3_57) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_57 * 32) + (ax2_0_3_57 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_57 * 32) + (ax2_0_3_57 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_57 * 32) + (ax2_0_3_57 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_57 * 32) + (ax2_0_3_57 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_57 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_57 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_57 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_57 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_57 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_57 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_57 * 32) + (ax2_0_3_57 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_57 * 32) + (ax2_0_3_57 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_57 * 32) + (ax2_0_3_57 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_57 * 32) + (ax2_0_3_57 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_57 * 32) + (ax2_0_3_57 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_57 * 32) + (ax2_0_3_57 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_57 * 32) + (ax2_0_3_57 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_57 * 32) + (ax2_0_3_57 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_57 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_57 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_57 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_57 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_57 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_57 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_57 * 32) + (ax2_0_3_57 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_57 * 32) + (ax2_0_3_57 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_57 * 32) + (ax2_0_3_57 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_57 * 32) + (ax2_0_3_57 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_122 = 0; ax0_ax1_fused_0_122 < 4; ++ax0_ax1_fused_0_122) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_122 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_122 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1952))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_123 = 0; ax0_ax1_fused_0_123 < 4; ++ax0_ax1_fused_0_123) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_123 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_123 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1952))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_58 = 0; ax3_0_1_58 < 2; ++ax3_0_1_58) {
    for (int ax0_0_116 = 0; ax0_0_116 < 4; ++ax0_0_116) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_116 * 512)) + (ax3_0_1_58 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_116 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_116 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_116 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_116 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_117 = 0; ax0_0_117 < 4; ++ax0_0_117) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_117 * 512)) + (ax3_0_1_58 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_117 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_117 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_117 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_117 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_58 = 0; ax1_0_3_58 < 4; ++ax1_0_3_58) {
      for (int ax2_0_3_58 = 0; ax2_0_3_58 < 4; ++ax2_0_3_58) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_58 * 32) + (ax2_0_3_58 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_58 * 32) + (ax2_0_3_58 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_58 * 32) + (ax2_0_3_58 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_58 * 32) + (ax2_0_3_58 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_58 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_58 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_58 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_58 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_58 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_58 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_58 * 32) + (ax2_0_3_58 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_58 * 32) + (ax2_0_3_58 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_58 * 32) + (ax2_0_3_58 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_58 * 32) + (ax2_0_3_58 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_58 * 32) + (ax2_0_3_58 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_58 * 32) + (ax2_0_3_58 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_58 * 32) + (ax2_0_3_58 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_58 * 32) + (ax2_0_3_58 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_58 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_58 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_58 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_58 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_58 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_58 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_58 * 32) + (ax2_0_3_58 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_58 * 32) + (ax2_0_3_58 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_58 * 32) + (ax2_0_3_58 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_58 * 32) + (ax2_0_3_58 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_124 = 0; ax0_ax1_fused_0_124 < 4; ++ax0_ax1_fused_0_124) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_124 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_124 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1984))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_125 = 0; ax0_ax1_fused_0_125 < 4; ++ax0_ax1_fused_0_125) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_125 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_125 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 1984))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_59 = 0; ax3_0_1_59 < 2; ++ax3_0_1_59) {
    for (int ax0_0_118 = 0; ax0_0_118 < 4; ++ax0_0_118) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_118 * 512)) + (ax3_0_1_59 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_118 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_118 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_118 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_118 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_119 = 0; ax0_0_119 < 4; ++ax0_0_119) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_119 * 512)) + (ax3_0_1_59 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_119 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_119 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_119 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_119 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_59 = 0; ax1_0_3_59 < 4; ++ax1_0_3_59) {
      for (int ax2_0_3_59 = 0; ax2_0_3_59 < 4; ++ax2_0_3_59) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_59 * 32) + (ax2_0_3_59 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_59 * 32) + (ax2_0_3_59 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_59 * 32) + (ax2_0_3_59 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_59 * 32) + (ax2_0_3_59 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_59 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_59 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_59 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_59 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_59 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_59 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_59 * 32) + (ax2_0_3_59 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_59 * 32) + (ax2_0_3_59 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_59 * 32) + (ax2_0_3_59 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_59 * 32) + (ax2_0_3_59 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_59 * 32) + (ax2_0_3_59 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_59 * 32) + (ax2_0_3_59 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_59 * 32) + (ax2_0_3_59 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_59 * 32) + (ax2_0_3_59 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_59 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_59 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_59 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_59 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_59 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_59 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_59 * 32) + (ax2_0_3_59 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_59 * 32) + (ax2_0_3_59 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_59 * 32) + (ax2_0_3_59 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_59 * 32) + (ax2_0_3_59 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_126 = 0; ax0_ax1_fused_0_126 < 4; ++ax0_ax1_fused_0_126) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_126 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_126 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2016))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_127 = 0; ax0_ax1_fused_0_127 < 4; ++ax0_ax1_fused_0_127) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_127 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_127 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2016))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_60 = 0; ax3_0_1_60 < 2; ++ax3_0_1_60) {
    for (int ax0_0_120 = 0; ax0_0_120 < 4; ++ax0_0_120) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_120 * 512)) + (ax3_0_1_60 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_120 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_120 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_120 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_120 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_121 = 0; ax0_0_121 < 4; ++ax0_0_121) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_121 * 512)) + (ax3_0_1_60 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_121 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_121 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_121 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_121 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_60 = 0; ax1_0_3_60 < 4; ++ax1_0_3_60) {
      for (int ax2_0_3_60 = 0; ax2_0_3_60 < 4; ++ax2_0_3_60) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_60 * 32) + (ax2_0_3_60 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_60 * 32) + (ax2_0_3_60 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_60 * 32) + (ax2_0_3_60 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_60 * 32) + (ax2_0_3_60 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_60 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_60 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_60 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_60 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_60 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_60 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_60 * 32) + (ax2_0_3_60 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_60 * 32) + (ax2_0_3_60 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_60 * 32) + (ax2_0_3_60 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_60 * 32) + (ax2_0_3_60 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_60 * 32) + (ax2_0_3_60 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_60 * 32) + (ax2_0_3_60 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_60 * 32) + (ax2_0_3_60 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_60 * 32) + (ax2_0_3_60 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_60 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_60 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_60 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_60 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_60 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_60 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_60 * 32) + (ax2_0_3_60 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_60 * 32) + (ax2_0_3_60 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_60 * 32) + (ax2_0_3_60 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_60 * 32) + (ax2_0_3_60 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_128 = 0; ax0_ax1_fused_0_128 < 4; ++ax0_ax1_fused_0_128) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_128 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_128 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2048))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_129 = 0; ax0_ax1_fused_0_129 < 4; ++ax0_ax1_fused_0_129) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_129 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_129 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2048))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_61 = 0; ax3_0_1_61 < 2; ++ax3_0_1_61) {
    for (int ax0_0_122 = 0; ax0_0_122 < 4; ++ax0_0_122) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_122 * 512)) + (ax3_0_1_61 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_122 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_122 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_122 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_122 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_123 = 0; ax0_0_123 < 4; ++ax0_0_123) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_123 * 512)) + (ax3_0_1_61 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_123 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_123 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_123 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_123 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_61 = 0; ax1_0_3_61 < 4; ++ax1_0_3_61) {
      for (int ax2_0_3_61 = 0; ax2_0_3_61 < 4; ++ax2_0_3_61) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_61 * 32) + (ax2_0_3_61 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_61 * 32) + (ax2_0_3_61 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_61 * 32) + (ax2_0_3_61 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_61 * 32) + (ax2_0_3_61 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_61 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_61 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_61 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_61 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_61 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_61 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_61 * 32) + (ax2_0_3_61 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_61 * 32) + (ax2_0_3_61 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_61 * 32) + (ax2_0_3_61 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_61 * 32) + (ax2_0_3_61 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_61 * 32) + (ax2_0_3_61 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_61 * 32) + (ax2_0_3_61 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_61 * 32) + (ax2_0_3_61 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_61 * 32) + (ax2_0_3_61 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_61 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_61 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_61 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_61 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_61 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_61 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_61 * 32) + (ax2_0_3_61 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_61 * 32) + (ax2_0_3_61 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_61 * 32) + (ax2_0_3_61 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_61 * 32) + (ax2_0_3_61 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_130 = 0; ax0_ax1_fused_0_130 < 4; ++ax0_ax1_fused_0_130) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_130 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_130 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2080))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_131 = 0; ax0_ax1_fused_0_131 < 4; ++ax0_ax1_fused_0_131) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_131 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_131 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2080))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_62 = 0; ax3_0_1_62 < 2; ++ax3_0_1_62) {
    for (int ax0_0_124 = 0; ax0_0_124 < 4; ++ax0_0_124) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_124 * 512)) + (ax3_0_1_62 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_124 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_124 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_124 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_124 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_125 = 0; ax0_0_125 < 4; ++ax0_0_125) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_125 * 512)) + (ax3_0_1_62 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_125 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_125 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_125 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_125 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_62 = 0; ax1_0_3_62 < 4; ++ax1_0_3_62) {
      for (int ax2_0_3_62 = 0; ax2_0_3_62 < 4; ++ax2_0_3_62) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_62 * 32) + (ax2_0_3_62 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_62 * 32) + (ax2_0_3_62 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_62 * 32) + (ax2_0_3_62 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_62 * 32) + (ax2_0_3_62 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_62 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_62 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_62 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_62 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_62 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_62 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_62 * 32) + (ax2_0_3_62 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_62 * 32) + (ax2_0_3_62 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_62 * 32) + (ax2_0_3_62 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_62 * 32) + (ax2_0_3_62 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_62 * 32) + (ax2_0_3_62 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_62 * 32) + (ax2_0_3_62 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_62 * 32) + (ax2_0_3_62 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_62 * 32) + (ax2_0_3_62 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_62 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_62 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_62 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_62 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_62 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_62 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_62 * 32) + (ax2_0_3_62 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_62 * 32) + (ax2_0_3_62 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_62 * 32) + (ax2_0_3_62 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_62 * 32) + (ax2_0_3_62 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_132 = 0; ax0_ax1_fused_0_132 < 4; ++ax0_ax1_fused_0_132) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_132 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_132 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2112))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_133 = 0; ax0_ax1_fused_0_133 < 4; ++ax0_ax1_fused_0_133) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_133 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_133 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2112))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_63 = 0; ax3_0_1_63 < 2; ++ax3_0_1_63) {
    for (int ax0_0_126 = 0; ax0_0_126 < 4; ++ax0_0_126) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_126 * 512)) + (ax3_0_1_63 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_126 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_126 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_126 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_126 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_127 = 0; ax0_0_127 < 4; ++ax0_0_127) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_127 * 512)) + (ax3_0_1_63 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_127 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_127 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_127 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_127 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_63 = 0; ax1_0_3_63 < 4; ++ax1_0_3_63) {
      for (int ax2_0_3_63 = 0; ax2_0_3_63 < 4; ++ax2_0_3_63) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_63 * 32) + (ax2_0_3_63 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_63 * 32) + (ax2_0_3_63 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_63 * 32) + (ax2_0_3_63 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_63 * 32) + (ax2_0_3_63 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_63 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_63 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_63 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_63 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_63 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_63 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_63 * 32) + (ax2_0_3_63 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_63 * 32) + (ax2_0_3_63 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_63 * 32) + (ax2_0_3_63 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_63 * 32) + (ax2_0_3_63 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_63 * 32) + (ax2_0_3_63 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_63 * 32) + (ax2_0_3_63 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_63 * 32) + (ax2_0_3_63 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_63 * 32) + (ax2_0_3_63 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_63 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_63 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_63 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_63 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_63 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_63 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_63 * 32) + (ax2_0_3_63 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_63 * 32) + (ax2_0_3_63 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_63 * 32) + (ax2_0_3_63 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_63 * 32) + (ax2_0_3_63 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_134 = 0; ax0_ax1_fused_0_134 < 4; ++ax0_ax1_fused_0_134) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_134 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_134 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2144))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_135 = 0; ax0_ax1_fused_0_135 < 4; ++ax0_ax1_fused_0_135) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_135 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_135 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2144))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_64 = 0; ax3_0_1_64 < 2; ++ax3_0_1_64) {
    for (int ax0_0_128 = 0; ax0_0_128 < 4; ++ax0_0_128) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_128 * 512)) + (ax3_0_1_64 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_128 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_128 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_128 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_128 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_129 = 0; ax0_0_129 < 4; ++ax0_0_129) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_129 * 512)) + (ax3_0_1_64 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_129 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_129 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_129 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_129 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_64 = 0; ax1_0_3_64 < 4; ++ax1_0_3_64) {
      for (int ax2_0_3_64 = 0; ax2_0_3_64 < 4; ++ax2_0_3_64) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_64 * 32) + (ax2_0_3_64 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_64 * 32) + (ax2_0_3_64 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_64 * 32) + (ax2_0_3_64 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_64 * 32) + (ax2_0_3_64 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_64 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_64 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_64 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_64 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_64 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_64 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_64 * 32) + (ax2_0_3_64 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_64 * 32) + (ax2_0_3_64 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_64 * 32) + (ax2_0_3_64 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_64 * 32) + (ax2_0_3_64 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_64 * 32) + (ax2_0_3_64 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_64 * 32) + (ax2_0_3_64 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_64 * 32) + (ax2_0_3_64 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_64 * 32) + (ax2_0_3_64 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_64 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_64 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_64 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_64 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_64 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_64 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_64 * 32) + (ax2_0_3_64 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_64 * 32) + (ax2_0_3_64 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_64 * 32) + (ax2_0_3_64 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_64 * 32) + (ax2_0_3_64 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_136 = 0; ax0_ax1_fused_0_136 < 4; ++ax0_ax1_fused_0_136) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_136 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_136 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2176))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_137 = 0; ax0_ax1_fused_0_137 < 4; ++ax0_ax1_fused_0_137) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_137 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_137 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2176))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_65 = 0; ax3_0_1_65 < 2; ++ax3_0_1_65) {
    for (int ax0_0_130 = 0; ax0_0_130 < 4; ++ax0_0_130) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_130 * 512)) + (ax3_0_1_65 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_130 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_130 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_130 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_130 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_131 = 0; ax0_0_131 < 4; ++ax0_0_131) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_131 * 512)) + (ax3_0_1_65 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_131 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_131 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_131 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_131 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_65 = 0; ax1_0_3_65 < 4; ++ax1_0_3_65) {
      for (int ax2_0_3_65 = 0; ax2_0_3_65 < 4; ++ax2_0_3_65) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_65 * 32) + (ax2_0_3_65 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_65 * 32) + (ax2_0_3_65 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_65 * 32) + (ax2_0_3_65 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_65 * 32) + (ax2_0_3_65 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_65 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_65 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_65 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_65 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_65 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_65 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_65 * 32) + (ax2_0_3_65 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_65 * 32) + (ax2_0_3_65 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_65 * 32) + (ax2_0_3_65 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_65 * 32) + (ax2_0_3_65 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_65 * 32) + (ax2_0_3_65 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_65 * 32) + (ax2_0_3_65 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_65 * 32) + (ax2_0_3_65 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_65 * 32) + (ax2_0_3_65 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_65 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_65 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_65 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_65 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_65 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_65 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_65 * 32) + (ax2_0_3_65 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_65 * 32) + (ax2_0_3_65 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_65 * 32) + (ax2_0_3_65 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_65 * 32) + (ax2_0_3_65 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_138 = 0; ax0_ax1_fused_0_138 < 4; ++ax0_ax1_fused_0_138) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_138 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_138 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2208))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_139 = 0; ax0_ax1_fused_0_139 < 4; ++ax0_ax1_fused_0_139) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_139 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_139 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2208))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_66 = 0; ax3_0_1_66 < 2; ++ax3_0_1_66) {
    for (int ax0_0_132 = 0; ax0_0_132 < 4; ++ax0_0_132) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_132 * 512)) + (ax3_0_1_66 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_132 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_132 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_132 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_132 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_133 = 0; ax0_0_133 < 4; ++ax0_0_133) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_133 * 512)) + (ax3_0_1_66 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_133 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_133 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_133 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_133 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_66 = 0; ax1_0_3_66 < 4; ++ax1_0_3_66) {
      for (int ax2_0_3_66 = 0; ax2_0_3_66 < 4; ++ax2_0_3_66) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_66 * 32) + (ax2_0_3_66 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_66 * 32) + (ax2_0_3_66 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_66 * 32) + (ax2_0_3_66 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_66 * 32) + (ax2_0_3_66 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_66 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_66 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_66 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_66 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_66 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_66 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_66 * 32) + (ax2_0_3_66 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_66 * 32) + (ax2_0_3_66 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_66 * 32) + (ax2_0_3_66 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_66 * 32) + (ax2_0_3_66 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_66 * 32) + (ax2_0_3_66 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_66 * 32) + (ax2_0_3_66 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_66 * 32) + (ax2_0_3_66 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_66 * 32) + (ax2_0_3_66 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_66 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_66 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_66 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_66 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_66 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_66 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_66 * 32) + (ax2_0_3_66 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_66 * 32) + (ax2_0_3_66 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_66 * 32) + (ax2_0_3_66 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_66 * 32) + (ax2_0_3_66 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_140 = 0; ax0_ax1_fused_0_140 < 4; ++ax0_ax1_fused_0_140) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_140 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_140 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2240))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_141 = 0; ax0_ax1_fused_0_141 < 4; ++ax0_ax1_fused_0_141) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_141 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_141 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2240))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_67 = 0; ax3_0_1_67 < 2; ++ax3_0_1_67) {
    for (int ax0_0_134 = 0; ax0_0_134 < 4; ++ax0_0_134) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_134 * 512)) + (ax3_0_1_67 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_134 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_134 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_134 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_134 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_135 = 0; ax0_0_135 < 4; ++ax0_0_135) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_135 * 512)) + (ax3_0_1_67 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_135 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_135 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_135 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_135 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_67 = 0; ax1_0_3_67 < 4; ++ax1_0_3_67) {
      for (int ax2_0_3_67 = 0; ax2_0_3_67 < 4; ++ax2_0_3_67) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_67 * 32) + (ax2_0_3_67 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_67 * 32) + (ax2_0_3_67 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_67 * 32) + (ax2_0_3_67 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_67 * 32) + (ax2_0_3_67 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_67 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_67 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_67 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_67 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_67 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_67 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_67 * 32) + (ax2_0_3_67 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_67 * 32) + (ax2_0_3_67 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_67 * 32) + (ax2_0_3_67 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_67 * 32) + (ax2_0_3_67 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_67 * 32) + (ax2_0_3_67 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_67 * 32) + (ax2_0_3_67 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_67 * 32) + (ax2_0_3_67 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_67 * 32) + (ax2_0_3_67 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_67 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_67 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_67 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_67 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_67 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_67 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_67 * 32) + (ax2_0_3_67 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_67 * 32) + (ax2_0_3_67 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_67 * 32) + (ax2_0_3_67 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_67 * 32) + (ax2_0_3_67 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_142 = 0; ax0_ax1_fused_0_142 < 4; ++ax0_ax1_fused_0_142) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_142 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_142 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2272))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_143 = 0; ax0_ax1_fused_0_143 < 4; ++ax0_ax1_fused_0_143) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_143 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_143 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2272))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_68 = 0; ax3_0_1_68 < 2; ++ax3_0_1_68) {
    for (int ax0_0_136 = 0; ax0_0_136 < 4; ++ax0_0_136) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_136 * 512)) + (ax3_0_1_68 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_136 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_136 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_136 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_136 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_137 = 0; ax0_0_137 < 4; ++ax0_0_137) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_137 * 512)) + (ax3_0_1_68 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_137 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_137 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_137 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_137 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_68 = 0; ax1_0_3_68 < 4; ++ax1_0_3_68) {
      for (int ax2_0_3_68 = 0; ax2_0_3_68 < 4; ++ax2_0_3_68) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_68 * 32) + (ax2_0_3_68 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_68 * 32) + (ax2_0_3_68 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_68 * 32) + (ax2_0_3_68 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_68 * 32) + (ax2_0_3_68 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_68 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_68 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_68 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_68 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_68 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_68 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_68 * 32) + (ax2_0_3_68 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_68 * 32) + (ax2_0_3_68 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_68 * 32) + (ax2_0_3_68 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_68 * 32) + (ax2_0_3_68 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_68 * 32) + (ax2_0_3_68 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_68 * 32) + (ax2_0_3_68 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_68 * 32) + (ax2_0_3_68 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_68 * 32) + (ax2_0_3_68 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_68 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_68 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_68 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_68 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_68 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_68 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_68 * 32) + (ax2_0_3_68 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_68 * 32) + (ax2_0_3_68 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_68 * 32) + (ax2_0_3_68 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_68 * 32) + (ax2_0_3_68 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_144 = 0; ax0_ax1_fused_0_144 < 4; ++ax0_ax1_fused_0_144) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_144 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_144 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2304))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_145 = 0; ax0_ax1_fused_0_145 < 4; ++ax0_ax1_fused_0_145) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_145 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_145 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2304))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_69 = 0; ax3_0_1_69 < 2; ++ax3_0_1_69) {
    for (int ax0_0_138 = 0; ax0_0_138 < 4; ++ax0_0_138) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_138 * 512)) + (ax3_0_1_69 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_138 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_138 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_138 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_138 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_139 = 0; ax0_0_139 < 4; ++ax0_0_139) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_139 * 512)) + (ax3_0_1_69 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_139 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_139 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_139 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_139 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_69 = 0; ax1_0_3_69 < 4; ++ax1_0_3_69) {
      for (int ax2_0_3_69 = 0; ax2_0_3_69 < 4; ++ax2_0_3_69) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_69 * 32) + (ax2_0_3_69 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_69 * 32) + (ax2_0_3_69 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_69 * 32) + (ax2_0_3_69 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_69 * 32) + (ax2_0_3_69 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_69 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_69 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_69 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_69 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_69 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_69 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_69 * 32) + (ax2_0_3_69 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_69 * 32) + (ax2_0_3_69 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_69 * 32) + (ax2_0_3_69 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_69 * 32) + (ax2_0_3_69 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_69 * 32) + (ax2_0_3_69 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_69 * 32) + (ax2_0_3_69 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_69 * 32) + (ax2_0_3_69 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_69 * 32) + (ax2_0_3_69 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_69 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_69 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_69 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_69 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_69 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_69 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_69 * 32) + (ax2_0_3_69 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_69 * 32) + (ax2_0_3_69 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_69 * 32) + (ax2_0_3_69 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_69 * 32) + (ax2_0_3_69 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_146 = 0; ax0_ax1_fused_0_146 < 4; ++ax0_ax1_fused_0_146) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_146 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_146 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2336))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_147 = 0; ax0_ax1_fused_0_147 < 4; ++ax0_ax1_fused_0_147) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_147 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_147 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2336))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_70 = 0; ax3_0_1_70 < 2; ++ax3_0_1_70) {
    for (int ax0_0_140 = 0; ax0_0_140 < 4; ++ax0_0_140) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_140 * 512)) + (ax3_0_1_70 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_140 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_140 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_140 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_140 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_141 = 0; ax0_0_141 < 4; ++ax0_0_141) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_141 * 512)) + (ax3_0_1_70 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_141 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_141 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_141 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_141 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_70 = 0; ax1_0_3_70 < 4; ++ax1_0_3_70) {
      for (int ax2_0_3_70 = 0; ax2_0_3_70 < 4; ++ax2_0_3_70) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_70 * 32) + (ax2_0_3_70 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_70 * 32) + (ax2_0_3_70 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_70 * 32) + (ax2_0_3_70 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_70 * 32) + (ax2_0_3_70 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_70 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_70 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_70 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_70 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_70 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_70 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_70 * 32) + (ax2_0_3_70 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_70 * 32) + (ax2_0_3_70 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_70 * 32) + (ax2_0_3_70 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_70 * 32) + (ax2_0_3_70 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_70 * 32) + (ax2_0_3_70 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_70 * 32) + (ax2_0_3_70 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_70 * 32) + (ax2_0_3_70 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_70 * 32) + (ax2_0_3_70 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_70 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_70 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_70 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_70 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_70 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_70 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_70 * 32) + (ax2_0_3_70 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_70 * 32) + (ax2_0_3_70 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_70 * 32) + (ax2_0_3_70 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_70 * 32) + (ax2_0_3_70 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_148 = 0; ax0_ax1_fused_0_148 < 4; ++ax0_ax1_fused_0_148) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_148 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_148 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2368))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_149 = 0; ax0_ax1_fused_0_149 < 4; ++ax0_ax1_fused_0_149) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_149 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_149 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2368))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_71 = 0; ax3_0_1_71 < 2; ++ax3_0_1_71) {
    for (int ax0_0_142 = 0; ax0_0_142 < 4; ++ax0_0_142) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_142 * 512)) + (ax3_0_1_71 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_142 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_142 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_142 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_142 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_143 = 0; ax0_0_143 < 4; ++ax0_0_143) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_143 * 512)) + (ax3_0_1_71 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_143 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_143 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_143 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_143 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_71 = 0; ax1_0_3_71 < 4; ++ax1_0_3_71) {
      for (int ax2_0_3_71 = 0; ax2_0_3_71 < 4; ++ax2_0_3_71) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_71 * 32) + (ax2_0_3_71 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_71 * 32) + (ax2_0_3_71 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_71 * 32) + (ax2_0_3_71 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_71 * 32) + (ax2_0_3_71 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_71 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_71 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_71 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_71 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_71 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_71 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_71 * 32) + (ax2_0_3_71 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_71 * 32) + (ax2_0_3_71 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_71 * 32) + (ax2_0_3_71 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_71 * 32) + (ax2_0_3_71 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_71 * 32) + (ax2_0_3_71 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_71 * 32) + (ax2_0_3_71 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_71 * 32) + (ax2_0_3_71 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_71 * 32) + (ax2_0_3_71 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_71 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_71 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_71 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_71 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_71 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_71 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_71 * 32) + (ax2_0_3_71 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_71 * 32) + (ax2_0_3_71 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_71 * 32) + (ax2_0_3_71 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_71 * 32) + (ax2_0_3_71 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_150 = 0; ax0_ax1_fused_0_150 < 4; ++ax0_ax1_fused_0_150) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_150 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_150 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2400))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_151 = 0; ax0_ax1_fused_0_151 < 4; ++ax0_ax1_fused_0_151) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_151 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_151 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2400))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_72 = 0; ax3_0_1_72 < 2; ++ax3_0_1_72) {
    for (int ax0_0_144 = 0; ax0_0_144 < 4; ++ax0_0_144) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_144 * 512)) + (ax3_0_1_72 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_144 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_144 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_144 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_144 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_145 = 0; ax0_0_145 < 4; ++ax0_0_145) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_145 * 512)) + (ax3_0_1_72 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_145 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_145 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_145 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_145 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_72 = 0; ax1_0_3_72 < 4; ++ax1_0_3_72) {
      for (int ax2_0_3_72 = 0; ax2_0_3_72 < 4; ++ax2_0_3_72) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_72 * 32) + (ax2_0_3_72 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_72 * 32) + (ax2_0_3_72 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_72 * 32) + (ax2_0_3_72 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_72 * 32) + (ax2_0_3_72 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_72 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_72 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_72 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_72 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_72 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_72 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_72 * 32) + (ax2_0_3_72 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_72 * 32) + (ax2_0_3_72 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_72 * 32) + (ax2_0_3_72 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_72 * 32) + (ax2_0_3_72 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_72 * 32) + (ax2_0_3_72 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_72 * 32) + (ax2_0_3_72 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_72 * 32) + (ax2_0_3_72 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_72 * 32) + (ax2_0_3_72 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_72 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_72 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_72 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_72 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_72 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_72 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_72 * 32) + (ax2_0_3_72 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_72 * 32) + (ax2_0_3_72 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_72 * 32) + (ax2_0_3_72 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_72 * 32) + (ax2_0_3_72 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_152 = 0; ax0_ax1_fused_0_152 < 4; ++ax0_ax1_fused_0_152) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_152 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_152 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2432))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_153 = 0; ax0_ax1_fused_0_153 < 4; ++ax0_ax1_fused_0_153) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_153 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_153 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2432))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_73 = 0; ax3_0_1_73 < 2; ++ax3_0_1_73) {
    for (int ax0_0_146 = 0; ax0_0_146 < 4; ++ax0_0_146) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_146 * 512)) + (ax3_0_1_73 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_146 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_146 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_146 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_146 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_147 = 0; ax0_0_147 < 4; ++ax0_0_147) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_147 * 512)) + (ax3_0_1_73 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_147 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_147 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_147 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_147 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_73 = 0; ax1_0_3_73 < 4; ++ax1_0_3_73) {
      for (int ax2_0_3_73 = 0; ax2_0_3_73 < 4; ++ax2_0_3_73) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_73 * 32) + (ax2_0_3_73 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_73 * 32) + (ax2_0_3_73 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_73 * 32) + (ax2_0_3_73 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_73 * 32) + (ax2_0_3_73 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_73 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_73 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_73 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_73 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_73 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_73 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_73 * 32) + (ax2_0_3_73 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_73 * 32) + (ax2_0_3_73 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_73 * 32) + (ax2_0_3_73 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_73 * 32) + (ax2_0_3_73 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_73 * 32) + (ax2_0_3_73 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_73 * 32) + (ax2_0_3_73 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_73 * 32) + (ax2_0_3_73 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_73 * 32) + (ax2_0_3_73 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_73 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_73 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_73 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_73 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_73 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_73 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_73 * 32) + (ax2_0_3_73 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_73 * 32) + (ax2_0_3_73 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_73 * 32) + (ax2_0_3_73 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_73 * 32) + (ax2_0_3_73 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_154 = 0; ax0_ax1_fused_0_154 < 4; ++ax0_ax1_fused_0_154) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_154 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_154 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2464))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_155 = 0; ax0_ax1_fused_0_155 < 4; ++ax0_ax1_fused_0_155) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_155 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_155 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2464))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_74 = 0; ax3_0_1_74 < 2; ++ax3_0_1_74) {
    for (int ax0_0_148 = 0; ax0_0_148 < 4; ++ax0_0_148) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_148 * 512)) + (ax3_0_1_74 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_148 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_148 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_148 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_148 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_149 = 0; ax0_0_149 < 4; ++ax0_0_149) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_149 * 512)) + (ax3_0_1_74 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_149 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_149 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_149 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_149 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_74 = 0; ax1_0_3_74 < 4; ++ax1_0_3_74) {
      for (int ax2_0_3_74 = 0; ax2_0_3_74 < 4; ++ax2_0_3_74) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_74 * 32) + (ax2_0_3_74 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_74 * 32) + (ax2_0_3_74 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_74 * 32) + (ax2_0_3_74 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_74 * 32) + (ax2_0_3_74 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_74 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_74 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_74 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_74 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_74 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_74 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_74 * 32) + (ax2_0_3_74 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_74 * 32) + (ax2_0_3_74 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_74 * 32) + (ax2_0_3_74 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_74 * 32) + (ax2_0_3_74 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_74 * 32) + (ax2_0_3_74 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_74 * 32) + (ax2_0_3_74 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_74 * 32) + (ax2_0_3_74 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_74 * 32) + (ax2_0_3_74 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_74 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_74 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_74 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_74 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_74 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_74 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_74 * 32) + (ax2_0_3_74 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_74 * 32) + (ax2_0_3_74 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_74 * 32) + (ax2_0_3_74 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_74 * 32) + (ax2_0_3_74 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_156 = 0; ax0_ax1_fused_0_156 < 4; ++ax0_ax1_fused_0_156) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_156 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_156 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2496))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_157 = 0; ax0_ax1_fused_0_157 < 4; ++ax0_ax1_fused_0_157) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_157 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_157 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2496))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_75 = 0; ax3_0_1_75 < 2; ++ax3_0_1_75) {
    for (int ax0_0_150 = 0; ax0_0_150 < 4; ++ax0_0_150) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_150 * 512)) + (ax3_0_1_75 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_150 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_150 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_150 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_150 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_151 = 0; ax0_0_151 < 4; ++ax0_0_151) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_151 * 512)) + (ax3_0_1_75 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_151 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_151 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_151 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_151 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_75 = 0; ax1_0_3_75 < 4; ++ax1_0_3_75) {
      for (int ax2_0_3_75 = 0; ax2_0_3_75 < 4; ++ax2_0_3_75) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_75 * 32) + (ax2_0_3_75 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_75 * 32) + (ax2_0_3_75 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_75 * 32) + (ax2_0_3_75 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_75 * 32) + (ax2_0_3_75 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_75 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_75 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_75 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_75 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_75 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_75 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_75 * 32) + (ax2_0_3_75 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_75 * 32) + (ax2_0_3_75 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_75 * 32) + (ax2_0_3_75 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_75 * 32) + (ax2_0_3_75 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_75 * 32) + (ax2_0_3_75 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_75 * 32) + (ax2_0_3_75 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_75 * 32) + (ax2_0_3_75 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_75 * 32) + (ax2_0_3_75 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_75 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_75 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_75 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_75 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_75 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_75 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_75 * 32) + (ax2_0_3_75 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_75 * 32) + (ax2_0_3_75 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_75 * 32) + (ax2_0_3_75 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_75 * 32) + (ax2_0_3_75 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_158 = 0; ax0_ax1_fused_0_158 < 4; ++ax0_ax1_fused_0_158) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_158 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_158 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2528))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_159 = 0; ax0_ax1_fused_0_159 < 4; ++ax0_ax1_fused_0_159) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_159 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_159 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2528))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_76 = 0; ax3_0_1_76 < 2; ++ax3_0_1_76) {
    for (int ax0_0_152 = 0; ax0_0_152 < 4; ++ax0_0_152) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_152 * 512)) + (ax3_0_1_76 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_152 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_152 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_152 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_152 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_153 = 0; ax0_0_153 < 4; ++ax0_0_153) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_153 * 512)) + (ax3_0_1_76 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_153 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_153 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_153 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_153 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_76 = 0; ax1_0_3_76 < 4; ++ax1_0_3_76) {
      for (int ax2_0_3_76 = 0; ax2_0_3_76 < 4; ++ax2_0_3_76) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_76 * 32) + (ax2_0_3_76 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_76 * 32) + (ax2_0_3_76 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_76 * 32) + (ax2_0_3_76 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_76 * 32) + (ax2_0_3_76 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_76 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_76 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_76 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_76 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_76 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_76 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_76 * 32) + (ax2_0_3_76 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_76 * 32) + (ax2_0_3_76 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_76 * 32) + (ax2_0_3_76 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_76 * 32) + (ax2_0_3_76 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_76 * 32) + (ax2_0_3_76 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_76 * 32) + (ax2_0_3_76 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_76 * 32) + (ax2_0_3_76 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_76 * 32) + (ax2_0_3_76 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_76 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_76 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_76 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_76 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_76 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_76 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_76 * 32) + (ax2_0_3_76 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_76 * 32) + (ax2_0_3_76 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_76 * 32) + (ax2_0_3_76 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_76 * 32) + (ax2_0_3_76 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_160 = 0; ax0_ax1_fused_0_160 < 4; ++ax0_ax1_fused_0_160) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_160 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_160 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2560))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_161 = 0; ax0_ax1_fused_0_161 < 4; ++ax0_ax1_fused_0_161) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_161 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_161 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2560))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_77 = 0; ax3_0_1_77 < 2; ++ax3_0_1_77) {
    for (int ax0_0_154 = 0; ax0_0_154 < 4; ++ax0_0_154) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_154 * 512)) + (ax3_0_1_77 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_154 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_154 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_154 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_154 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_155 = 0; ax0_0_155 < 4; ++ax0_0_155) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_155 * 512)) + (ax3_0_1_77 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_155 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_155 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_155 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_155 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_77 = 0; ax1_0_3_77 < 4; ++ax1_0_3_77) {
      for (int ax2_0_3_77 = 0; ax2_0_3_77 < 4; ++ax2_0_3_77) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_77 * 32) + (ax2_0_3_77 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_77 * 32) + (ax2_0_3_77 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_77 * 32) + (ax2_0_3_77 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_77 * 32) + (ax2_0_3_77 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_77 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_77 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_77 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_77 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_77 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_77 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_77 * 32) + (ax2_0_3_77 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_77 * 32) + (ax2_0_3_77 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_77 * 32) + (ax2_0_3_77 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_77 * 32) + (ax2_0_3_77 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_77 * 32) + (ax2_0_3_77 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_77 * 32) + (ax2_0_3_77 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_77 * 32) + (ax2_0_3_77 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_77 * 32) + (ax2_0_3_77 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_77 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_77 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_77 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_77 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_77 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_77 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_77 * 32) + (ax2_0_3_77 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_77 * 32) + (ax2_0_3_77 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_77 * 32) + (ax2_0_3_77 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_77 * 32) + (ax2_0_3_77 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_162 = 0; ax0_ax1_fused_0_162 < 4; ++ax0_ax1_fused_0_162) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_162 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_162 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2592))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_163 = 0; ax0_ax1_fused_0_163 < 4; ++ax0_ax1_fused_0_163) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_163 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_163 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2592))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_78 = 0; ax3_0_1_78 < 2; ++ax3_0_1_78) {
    for (int ax0_0_156 = 0; ax0_0_156 < 4; ++ax0_0_156) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_156 * 512)) + (ax3_0_1_78 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_156 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_156 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_156 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_156 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_157 = 0; ax0_0_157 < 4; ++ax0_0_157) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_157 * 512)) + (ax3_0_1_78 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_157 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_157 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_157 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_157 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_78 = 0; ax1_0_3_78 < 4; ++ax1_0_3_78) {
      for (int ax2_0_3_78 = 0; ax2_0_3_78 < 4; ++ax2_0_3_78) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_78 * 32) + (ax2_0_3_78 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_78 * 32) + (ax2_0_3_78 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_78 * 32) + (ax2_0_3_78 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_78 * 32) + (ax2_0_3_78 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_78 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_78 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_78 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_78 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_78 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_78 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_78 * 32) + (ax2_0_3_78 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_78 * 32) + (ax2_0_3_78 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_78 * 32) + (ax2_0_3_78 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_78 * 32) + (ax2_0_3_78 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_78 * 32) + (ax2_0_3_78 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_78 * 32) + (ax2_0_3_78 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_78 * 32) + (ax2_0_3_78 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_78 * 32) + (ax2_0_3_78 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_78 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_78 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_78 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_78 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_78 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_78 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_78 * 32) + (ax2_0_3_78 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_78 * 32) + (ax2_0_3_78 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_78 * 32) + (ax2_0_3_78 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_78 * 32) + (ax2_0_3_78 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_164 = 0; ax0_ax1_fused_0_164 < 4; ++ax0_ax1_fused_0_164) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_164 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_164 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2624))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_165 = 0; ax0_ax1_fused_0_165 < 4; ++ax0_ax1_fused_0_165) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_165 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_165 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2624))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_79 = 0; ax3_0_1_79 < 2; ++ax3_0_1_79) {
    for (int ax0_0_158 = 0; ax0_0_158 < 4; ++ax0_0_158) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_158 * 512)) + (ax3_0_1_79 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_158 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_158 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_158 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_158 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_159 = 0; ax0_0_159 < 4; ++ax0_0_159) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_159 * 512)) + (ax3_0_1_79 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_159 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_159 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_159 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_159 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_79 = 0; ax1_0_3_79 < 4; ++ax1_0_3_79) {
      for (int ax2_0_3_79 = 0; ax2_0_3_79 < 4; ++ax2_0_3_79) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_79 * 32) + (ax2_0_3_79 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_79 * 32) + (ax2_0_3_79 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_79 * 32) + (ax2_0_3_79 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_79 * 32) + (ax2_0_3_79 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_79 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_79 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_79 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_79 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_79 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_79 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_79 * 32) + (ax2_0_3_79 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_79 * 32) + (ax2_0_3_79 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_79 * 32) + (ax2_0_3_79 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_79 * 32) + (ax2_0_3_79 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_79 * 32) + (ax2_0_3_79 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_79 * 32) + (ax2_0_3_79 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_79 * 32) + (ax2_0_3_79 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_79 * 32) + (ax2_0_3_79 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_79 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_79 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_79 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_79 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_79 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_79 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_79 * 32) + (ax2_0_3_79 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_79 * 32) + (ax2_0_3_79 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_79 * 32) + (ax2_0_3_79 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_79 * 32) + (ax2_0_3_79 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_166 = 0; ax0_ax1_fused_0_166 < 4; ++ax0_ax1_fused_0_166) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_166 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_166 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2656))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_167 = 0; ax0_ax1_fused_0_167 < 4; ++ax0_ax1_fused_0_167) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_167 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_167 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2656))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_80 = 0; ax3_0_1_80 < 2; ++ax3_0_1_80) {
    for (int ax0_0_160 = 0; ax0_0_160 < 4; ++ax0_0_160) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_160 * 512)) + (ax3_0_1_80 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_160 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_160 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_160 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_160 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_161 = 0; ax0_0_161 < 4; ++ax0_0_161) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_161 * 512)) + (ax3_0_1_80 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_161 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_161 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_161 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_161 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_80 = 0; ax1_0_3_80 < 4; ++ax1_0_3_80) {
      for (int ax2_0_3_80 = 0; ax2_0_3_80 < 4; ++ax2_0_3_80) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_80 * 32) + (ax2_0_3_80 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_80 * 32) + (ax2_0_3_80 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_80 * 32) + (ax2_0_3_80 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_80 * 32) + (ax2_0_3_80 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_80 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_80 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_80 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_80 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_80 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_80 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_80 * 32) + (ax2_0_3_80 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_80 * 32) + (ax2_0_3_80 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_80 * 32) + (ax2_0_3_80 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_80 * 32) + (ax2_0_3_80 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_80 * 32) + (ax2_0_3_80 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_80 * 32) + (ax2_0_3_80 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_80 * 32) + (ax2_0_3_80 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_80 * 32) + (ax2_0_3_80 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_80 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_80 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_80 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_80 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_80 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_80 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_80 * 32) + (ax2_0_3_80 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_80 * 32) + (ax2_0_3_80 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_80 * 32) + (ax2_0_3_80 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_80 * 32) + (ax2_0_3_80 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_168 = 0; ax0_ax1_fused_0_168 < 4; ++ax0_ax1_fused_0_168) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_168 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_168 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2688))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_169 = 0; ax0_ax1_fused_0_169 < 4; ++ax0_ax1_fused_0_169) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_169 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_169 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2688))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_81 = 0; ax3_0_1_81 < 2; ++ax3_0_1_81) {
    for (int ax0_0_162 = 0; ax0_0_162 < 4; ++ax0_0_162) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_162 * 512)) + (ax3_0_1_81 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_162 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_162 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_162 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_162 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_163 = 0; ax0_0_163 < 4; ++ax0_0_163) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_163 * 512)) + (ax3_0_1_81 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_163 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_163 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_163 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_163 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_81 = 0; ax1_0_3_81 < 4; ++ax1_0_3_81) {
      for (int ax2_0_3_81 = 0; ax2_0_3_81 < 4; ++ax2_0_3_81) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_81 * 32) + (ax2_0_3_81 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_81 * 32) + (ax2_0_3_81 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_81 * 32) + (ax2_0_3_81 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_81 * 32) + (ax2_0_3_81 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_81 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_81 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_81 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_81 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_81 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_81 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_81 * 32) + (ax2_0_3_81 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_81 * 32) + (ax2_0_3_81 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_81 * 32) + (ax2_0_3_81 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_81 * 32) + (ax2_0_3_81 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_81 * 32) + (ax2_0_3_81 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_81 * 32) + (ax2_0_3_81 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_81 * 32) + (ax2_0_3_81 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_81 * 32) + (ax2_0_3_81 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_81 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_81 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_81 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_81 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_81 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_81 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_81 * 32) + (ax2_0_3_81 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_81 * 32) + (ax2_0_3_81 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_81 * 32) + (ax2_0_3_81 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_81 * 32) + (ax2_0_3_81 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_170 = 0; ax0_ax1_fused_0_170 < 4; ++ax0_ax1_fused_0_170) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_170 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_170 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2720))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_171 = 0; ax0_ax1_fused_0_171 < 4; ++ax0_ax1_fused_0_171) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_171 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_171 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2720))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_82 = 0; ax3_0_1_82 < 2; ++ax3_0_1_82) {
    for (int ax0_0_164 = 0; ax0_0_164 < 4; ++ax0_0_164) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_164 * 512)) + (ax3_0_1_82 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_164 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_164 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_164 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_164 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_165 = 0; ax0_0_165 < 4; ++ax0_0_165) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_165 * 512)) + (ax3_0_1_82 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_165 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_165 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_165 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_165 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_82 = 0; ax1_0_3_82 < 4; ++ax1_0_3_82) {
      for (int ax2_0_3_82 = 0; ax2_0_3_82 < 4; ++ax2_0_3_82) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_82 * 32) + (ax2_0_3_82 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_82 * 32) + (ax2_0_3_82 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_82 * 32) + (ax2_0_3_82 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_82 * 32) + (ax2_0_3_82 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_82 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_82 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_82 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_82 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_82 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_82 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_82 * 32) + (ax2_0_3_82 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_82 * 32) + (ax2_0_3_82 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_82 * 32) + (ax2_0_3_82 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_82 * 32) + (ax2_0_3_82 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_82 * 32) + (ax2_0_3_82 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_82 * 32) + (ax2_0_3_82 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_82 * 32) + (ax2_0_3_82 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_82 * 32) + (ax2_0_3_82 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_82 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_82 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_82 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_82 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_82 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_82 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_82 * 32) + (ax2_0_3_82 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_82 * 32) + (ax2_0_3_82 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_82 * 32) + (ax2_0_3_82 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_82 * 32) + (ax2_0_3_82 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_172 = 0; ax0_ax1_fused_0_172 < 4; ++ax0_ax1_fused_0_172) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_172 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_172 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2752))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_173 = 0; ax0_ax1_fused_0_173 < 4; ++ax0_ax1_fused_0_173) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_173 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_173 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2752))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_83 = 0; ax3_0_1_83 < 2; ++ax3_0_1_83) {
    for (int ax0_0_166 = 0; ax0_0_166 < 4; ++ax0_0_166) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_166 * 512)) + (ax3_0_1_83 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_166 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_166 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_166 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_166 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_167 = 0; ax0_0_167 < 4; ++ax0_0_167) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_167 * 512)) + (ax3_0_1_83 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_167 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_167 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_167 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_167 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_83 = 0; ax1_0_3_83 < 4; ++ax1_0_3_83) {
      for (int ax2_0_3_83 = 0; ax2_0_3_83 < 4; ++ax2_0_3_83) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_83 * 32) + (ax2_0_3_83 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_83 * 32) + (ax2_0_3_83 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_83 * 32) + (ax2_0_3_83 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_83 * 32) + (ax2_0_3_83 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_83 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_83 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_83 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_83 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_83 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_83 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_83 * 32) + (ax2_0_3_83 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_83 * 32) + (ax2_0_3_83 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_83 * 32) + (ax2_0_3_83 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_83 * 32) + (ax2_0_3_83 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_83 * 32) + (ax2_0_3_83 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_83 * 32) + (ax2_0_3_83 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_83 * 32) + (ax2_0_3_83 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_83 * 32) + (ax2_0_3_83 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_83 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_83 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_83 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_83 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_83 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_83 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_83 * 32) + (ax2_0_3_83 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_83 * 32) + (ax2_0_3_83 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_83 * 32) + (ax2_0_3_83 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_83 * 32) + (ax2_0_3_83 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_174 = 0; ax0_ax1_fused_0_174 < 4; ++ax0_ax1_fused_0_174) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_174 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_174 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2784))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_175 = 0; ax0_ax1_fused_0_175 < 4; ++ax0_ax1_fused_0_175) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_175 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_175 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2784))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_84 = 0; ax3_0_1_84 < 2; ++ax3_0_1_84) {
    for (int ax0_0_168 = 0; ax0_0_168 < 4; ++ax0_0_168) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_168 * 512)) + (ax3_0_1_84 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_168 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_168 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_168 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_168 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_169 = 0; ax0_0_169 < 4; ++ax0_0_169) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_169 * 512)) + (ax3_0_1_84 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_169 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_169 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_169 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_169 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_84 = 0; ax1_0_3_84 < 4; ++ax1_0_3_84) {
      for (int ax2_0_3_84 = 0; ax2_0_3_84 < 4; ++ax2_0_3_84) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_84 * 32) + (ax2_0_3_84 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_84 * 32) + (ax2_0_3_84 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_84 * 32) + (ax2_0_3_84 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_84 * 32) + (ax2_0_3_84 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_84 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_84 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_84 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_84 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_84 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_84 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_84 * 32) + (ax2_0_3_84 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_84 * 32) + (ax2_0_3_84 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_84 * 32) + (ax2_0_3_84 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_84 * 32) + (ax2_0_3_84 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_84 * 32) + (ax2_0_3_84 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_84 * 32) + (ax2_0_3_84 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_84 * 32) + (ax2_0_3_84 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_84 * 32) + (ax2_0_3_84 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_84 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_84 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_84 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_84 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_84 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_84 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_84 * 32) + (ax2_0_3_84 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_84 * 32) + (ax2_0_3_84 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_84 * 32) + (ax2_0_3_84 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_84 * 32) + (ax2_0_3_84 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_176 = 0; ax0_ax1_fused_0_176 < 4; ++ax0_ax1_fused_0_176) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_176 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_176 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2816))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_177 = 0; ax0_ax1_fused_0_177 < 4; ++ax0_ax1_fused_0_177) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_177 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_177 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2816))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_85 = 0; ax3_0_1_85 < 2; ++ax3_0_1_85) {
    for (int ax0_0_170 = 0; ax0_0_170 < 4; ++ax0_0_170) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_170 * 512)) + (ax3_0_1_85 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_170 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_170 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_170 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_170 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_171 = 0; ax0_0_171 < 4; ++ax0_0_171) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_171 * 512)) + (ax3_0_1_85 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_171 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_171 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_171 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_171 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_85 = 0; ax1_0_3_85 < 4; ++ax1_0_3_85) {
      for (int ax2_0_3_85 = 0; ax2_0_3_85 < 4; ++ax2_0_3_85) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_85 * 32) + (ax2_0_3_85 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_85 * 32) + (ax2_0_3_85 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_85 * 32) + (ax2_0_3_85 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_85 * 32) + (ax2_0_3_85 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_85 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_85 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_85 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_85 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_85 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_85 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_85 * 32) + (ax2_0_3_85 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_85 * 32) + (ax2_0_3_85 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_85 * 32) + (ax2_0_3_85 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_85 * 32) + (ax2_0_3_85 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_85 * 32) + (ax2_0_3_85 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_85 * 32) + (ax2_0_3_85 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_85 * 32) + (ax2_0_3_85 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_85 * 32) + (ax2_0_3_85 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_85 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_85 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_85 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_85 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_85 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_85 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_85 * 32) + (ax2_0_3_85 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_85 * 32) + (ax2_0_3_85 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_85 * 32) + (ax2_0_3_85 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_85 * 32) + (ax2_0_3_85 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_178 = 0; ax0_ax1_fused_0_178 < 4; ++ax0_ax1_fused_0_178) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_178 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_178 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2848))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_179 = 0; ax0_ax1_fused_0_179 < 4; ++ax0_ax1_fused_0_179) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_179 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_179 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2848))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_86 = 0; ax3_0_1_86 < 2; ++ax3_0_1_86) {
    for (int ax0_0_172 = 0; ax0_0_172 < 4; ++ax0_0_172) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_172 * 512)) + (ax3_0_1_86 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_172 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_172 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_172 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_172 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_173 = 0; ax0_0_173 < 4; ++ax0_0_173) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_173 * 512)) + (ax3_0_1_86 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_173 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_173 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_173 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_173 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_86 = 0; ax1_0_3_86 < 4; ++ax1_0_3_86) {
      for (int ax2_0_3_86 = 0; ax2_0_3_86 < 4; ++ax2_0_3_86) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_86 * 32) + (ax2_0_3_86 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_86 * 32) + (ax2_0_3_86 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_86 * 32) + (ax2_0_3_86 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_86 * 32) + (ax2_0_3_86 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_86 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_86 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_86 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_86 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_86 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_86 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_86 * 32) + (ax2_0_3_86 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_86 * 32) + (ax2_0_3_86 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_86 * 32) + (ax2_0_3_86 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_86 * 32) + (ax2_0_3_86 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_86 * 32) + (ax2_0_3_86 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_86 * 32) + (ax2_0_3_86 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_86 * 32) + (ax2_0_3_86 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_86 * 32) + (ax2_0_3_86 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_86 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_86 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_86 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_86 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_86 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_86 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_86 * 32) + (ax2_0_3_86 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_86 * 32) + (ax2_0_3_86 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_86 * 32) + (ax2_0_3_86 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_86 * 32) + (ax2_0_3_86 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_180 = 0; ax0_ax1_fused_0_180 < 4; ++ax0_ax1_fused_0_180) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_180 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_180 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2880))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_181 = 0; ax0_ax1_fused_0_181 < 4; ++ax0_ax1_fused_0_181) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_181 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_181 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2880))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_87 = 0; ax3_0_1_87 < 2; ++ax3_0_1_87) {
    for (int ax0_0_174 = 0; ax0_0_174 < 4; ++ax0_0_174) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_174 * 512)) + (ax3_0_1_87 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_174 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_174 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_174 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_174 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_175 = 0; ax0_0_175 < 4; ++ax0_0_175) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_175 * 512)) + (ax3_0_1_87 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_175 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_175 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_175 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_175 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_87 = 0; ax1_0_3_87 < 4; ++ax1_0_3_87) {
      for (int ax2_0_3_87 = 0; ax2_0_3_87 < 4; ++ax2_0_3_87) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_87 * 32) + (ax2_0_3_87 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_87 * 32) + (ax2_0_3_87 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_87 * 32) + (ax2_0_3_87 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_87 * 32) + (ax2_0_3_87 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_87 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_87 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_87 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_87 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_87 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_87 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_87 * 32) + (ax2_0_3_87 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_87 * 32) + (ax2_0_3_87 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_87 * 32) + (ax2_0_3_87 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_87 * 32) + (ax2_0_3_87 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_87 * 32) + (ax2_0_3_87 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_87 * 32) + (ax2_0_3_87 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_87 * 32) + (ax2_0_3_87 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_87 * 32) + (ax2_0_3_87 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_87 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_87 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_87 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_87 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_87 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_87 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_87 * 32) + (ax2_0_3_87 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_87 * 32) + (ax2_0_3_87 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_87 * 32) + (ax2_0_3_87 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_87 * 32) + (ax2_0_3_87 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_182 = 0; ax0_ax1_fused_0_182 < 4; ++ax0_ax1_fused_0_182) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_182 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_182 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2912))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_183 = 0; ax0_ax1_fused_0_183 < 4; ++ax0_ax1_fused_0_183) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_183 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_183 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2912))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_88 = 0; ax3_0_1_88 < 2; ++ax3_0_1_88) {
    for (int ax0_0_176 = 0; ax0_0_176 < 4; ++ax0_0_176) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_176 * 512)) + (ax3_0_1_88 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_176 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_176 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_176 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_176 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_177 = 0; ax0_0_177 < 4; ++ax0_0_177) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_177 * 512)) + (ax3_0_1_88 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_177 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_177 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_177 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_177 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_88 = 0; ax1_0_3_88 < 4; ++ax1_0_3_88) {
      for (int ax2_0_3_88 = 0; ax2_0_3_88 < 4; ++ax2_0_3_88) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_88 * 32) + (ax2_0_3_88 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_88 * 32) + (ax2_0_3_88 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_88 * 32) + (ax2_0_3_88 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_88 * 32) + (ax2_0_3_88 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_88 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_88 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_88 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_88 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_88 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_88 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_88 * 32) + (ax2_0_3_88 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_88 * 32) + (ax2_0_3_88 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_88 * 32) + (ax2_0_3_88 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_88 * 32) + (ax2_0_3_88 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_88 * 32) + (ax2_0_3_88 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_88 * 32) + (ax2_0_3_88 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_88 * 32) + (ax2_0_3_88 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_88 * 32) + (ax2_0_3_88 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_88 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_88 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_88 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_88 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_88 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_88 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_88 * 32) + (ax2_0_3_88 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_88 * 32) + (ax2_0_3_88 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_88 * 32) + (ax2_0_3_88 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_88 * 32) + (ax2_0_3_88 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_184 = 0; ax0_ax1_fused_0_184 < 4; ++ax0_ax1_fused_0_184) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_184 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_184 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2944))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_185 = 0; ax0_ax1_fused_0_185 < 4; ++ax0_ax1_fused_0_185) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_185 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_185 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2944))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_89 = 0; ax3_0_1_89 < 2; ++ax3_0_1_89) {
    for (int ax0_0_178 = 0; ax0_0_178 < 4; ++ax0_0_178) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_178 * 512)) + (ax3_0_1_89 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_178 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_178 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_178 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_178 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_179 = 0; ax0_0_179 < 4; ++ax0_0_179) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_179 * 512)) + (ax3_0_1_89 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_179 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_179 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_179 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_179 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_89 = 0; ax1_0_3_89 < 4; ++ax1_0_3_89) {
      for (int ax2_0_3_89 = 0; ax2_0_3_89 < 4; ++ax2_0_3_89) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_89 * 32) + (ax2_0_3_89 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_89 * 32) + (ax2_0_3_89 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_89 * 32) + (ax2_0_3_89 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_89 * 32) + (ax2_0_3_89 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_89 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_89 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_89 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_89 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_89 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_89 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_89 * 32) + (ax2_0_3_89 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_89 * 32) + (ax2_0_3_89 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_89 * 32) + (ax2_0_3_89 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_89 * 32) + (ax2_0_3_89 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_89 * 32) + (ax2_0_3_89 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_89 * 32) + (ax2_0_3_89 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_89 * 32) + (ax2_0_3_89 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_89 * 32) + (ax2_0_3_89 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_89 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_89 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_89 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_89 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_89 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_89 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_89 * 32) + (ax2_0_3_89 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_89 * 32) + (ax2_0_3_89 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_89 * 32) + (ax2_0_3_89 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_89 * 32) + (ax2_0_3_89 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_186 = 0; ax0_ax1_fused_0_186 < 4; ++ax0_ax1_fused_0_186) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_186 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_186 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2976))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_187 = 0; ax0_ax1_fused_0_187 < 4; ++ax0_ax1_fused_0_187) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_187 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_187 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 2976))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_90 = 0; ax3_0_1_90 < 2; ++ax3_0_1_90) {
    for (int ax0_0_180 = 0; ax0_0_180 < 4; ++ax0_0_180) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_180 * 512)) + (ax3_0_1_90 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_180 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_180 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_180 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_180 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_181 = 0; ax0_0_181 < 4; ++ax0_0_181) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_181 * 512)) + (ax3_0_1_90 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_181 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_181 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_181 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_181 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_90 = 0; ax1_0_3_90 < 4; ++ax1_0_3_90) {
      for (int ax2_0_3_90 = 0; ax2_0_3_90 < 4; ++ax2_0_3_90) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_90 * 32) + (ax2_0_3_90 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_90 * 32) + (ax2_0_3_90 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_90 * 32) + (ax2_0_3_90 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_90 * 32) + (ax2_0_3_90 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_90 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_90 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_90 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_90 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_90 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_90 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_90 * 32) + (ax2_0_3_90 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_90 * 32) + (ax2_0_3_90 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_90 * 32) + (ax2_0_3_90 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_90 * 32) + (ax2_0_3_90 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_90 * 32) + (ax2_0_3_90 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_90 * 32) + (ax2_0_3_90 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_90 * 32) + (ax2_0_3_90 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_90 * 32) + (ax2_0_3_90 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_90 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_90 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_90 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_90 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_90 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_90 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_90 * 32) + (ax2_0_3_90 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_90 * 32) + (ax2_0_3_90 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_90 * 32) + (ax2_0_3_90 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_90 * 32) + (ax2_0_3_90 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_188 = 0; ax0_ax1_fused_0_188 < 4; ++ax0_ax1_fused_0_188) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_188 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_188 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3008))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_189 = 0; ax0_ax1_fused_0_189 < 4; ++ax0_ax1_fused_0_189) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_189 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_189 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3008))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_91 = 0; ax3_0_1_91 < 2; ++ax3_0_1_91) {
    for (int ax0_0_182 = 0; ax0_0_182 < 4; ++ax0_0_182) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_182 * 512)) + (ax3_0_1_91 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_182 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_182 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_182 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_182 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_183 = 0; ax0_0_183 < 4; ++ax0_0_183) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_183 * 512)) + (ax3_0_1_91 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_183 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_183 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_183 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_183 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_91 = 0; ax1_0_3_91 < 4; ++ax1_0_3_91) {
      for (int ax2_0_3_91 = 0; ax2_0_3_91 < 4; ++ax2_0_3_91) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_91 * 32) + (ax2_0_3_91 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_91 * 32) + (ax2_0_3_91 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_91 * 32) + (ax2_0_3_91 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_91 * 32) + (ax2_0_3_91 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_91 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_91 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_91 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_91 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_91 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_91 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_91 * 32) + (ax2_0_3_91 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_91 * 32) + (ax2_0_3_91 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_91 * 32) + (ax2_0_3_91 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_91 * 32) + (ax2_0_3_91 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_91 * 32) + (ax2_0_3_91 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_91 * 32) + (ax2_0_3_91 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_91 * 32) + (ax2_0_3_91 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_91 * 32) + (ax2_0_3_91 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_91 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_91 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_91 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_91 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_91 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_91 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_91 * 32) + (ax2_0_3_91 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_91 * 32) + (ax2_0_3_91 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_91 * 32) + (ax2_0_3_91 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_91 * 32) + (ax2_0_3_91 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_190 = 0; ax0_ax1_fused_0_190 < 4; ++ax0_ax1_fused_0_190) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_190 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_190 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3040))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_191 = 0; ax0_ax1_fused_0_191 < 4; ++ax0_ax1_fused_0_191) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_191 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_191 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3040))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_92 = 0; ax3_0_1_92 < 2; ++ax3_0_1_92) {
    for (int ax0_0_184 = 0; ax0_0_184 < 4; ++ax0_0_184) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_184 * 512)) + (ax3_0_1_92 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_184 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_184 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_184 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_184 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_185 = 0; ax0_0_185 < 4; ++ax0_0_185) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_185 * 512)) + (ax3_0_1_92 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_185 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_185 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_185 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_185 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_92 = 0; ax1_0_3_92 < 4; ++ax1_0_3_92) {
      for (int ax2_0_3_92 = 0; ax2_0_3_92 < 4; ++ax2_0_3_92) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_92 * 32) + (ax2_0_3_92 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_92 * 32) + (ax2_0_3_92 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_92 * 32) + (ax2_0_3_92 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_92 * 32) + (ax2_0_3_92 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_92 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_92 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_92 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_92 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_92 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_92 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_92 * 32) + (ax2_0_3_92 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_92 * 32) + (ax2_0_3_92 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_92 * 32) + (ax2_0_3_92 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_92 * 32) + (ax2_0_3_92 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_92 * 32) + (ax2_0_3_92 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_92 * 32) + (ax2_0_3_92 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_92 * 32) + (ax2_0_3_92 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_92 * 32) + (ax2_0_3_92 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_92 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_92 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_92 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_92 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_92 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_92 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_92 * 32) + (ax2_0_3_92 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_92 * 32) + (ax2_0_3_92 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_92 * 32) + (ax2_0_3_92 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_92 * 32) + (ax2_0_3_92 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_192 = 0; ax0_ax1_fused_0_192 < 4; ++ax0_ax1_fused_0_192) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_192 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_192 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3072))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_193 = 0; ax0_ax1_fused_0_193 < 4; ++ax0_ax1_fused_0_193) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_193 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_193 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3072))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_93 = 0; ax3_0_1_93 < 2; ++ax3_0_1_93) {
    for (int ax0_0_186 = 0; ax0_0_186 < 4; ++ax0_0_186) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_186 * 512)) + (ax3_0_1_93 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_186 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_186 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_186 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_186 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_187 = 0; ax0_0_187 < 4; ++ax0_0_187) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_187 * 512)) + (ax3_0_1_93 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_187 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_187 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_187 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_187 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_93 = 0; ax1_0_3_93 < 4; ++ax1_0_3_93) {
      for (int ax2_0_3_93 = 0; ax2_0_3_93 < 4; ++ax2_0_3_93) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_93 * 32) + (ax2_0_3_93 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_93 * 32) + (ax2_0_3_93 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_93 * 32) + (ax2_0_3_93 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_93 * 32) + (ax2_0_3_93 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_93 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_93 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_93 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_93 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_93 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_93 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_93 * 32) + (ax2_0_3_93 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_93 * 32) + (ax2_0_3_93 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_93 * 32) + (ax2_0_3_93 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_93 * 32) + (ax2_0_3_93 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_93 * 32) + (ax2_0_3_93 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_93 * 32) + (ax2_0_3_93 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_93 * 32) + (ax2_0_3_93 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_93 * 32) + (ax2_0_3_93 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_93 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_93 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_93 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_93 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_93 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_93 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_93 * 32) + (ax2_0_3_93 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_93 * 32) + (ax2_0_3_93 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_93 * 32) + (ax2_0_3_93 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_93 * 32) + (ax2_0_3_93 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_194 = 0; ax0_ax1_fused_0_194 < 4; ++ax0_ax1_fused_0_194) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_194 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_194 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3104))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_195 = 0; ax0_ax1_fused_0_195 < 4; ++ax0_ax1_fused_0_195) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_195 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_195 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3104))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_94 = 0; ax3_0_1_94 < 2; ++ax3_0_1_94) {
    for (int ax0_0_188 = 0; ax0_0_188 < 4; ++ax0_0_188) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_188 * 512)) + (ax3_0_1_94 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_188 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_188 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_188 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_188 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_189 = 0; ax0_0_189 < 4; ++ax0_0_189) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_189 * 512)) + (ax3_0_1_94 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_189 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_189 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_189 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_189 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_94 = 0; ax1_0_3_94 < 4; ++ax1_0_3_94) {
      for (int ax2_0_3_94 = 0; ax2_0_3_94 < 4; ++ax2_0_3_94) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_94 * 32) + (ax2_0_3_94 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_94 * 32) + (ax2_0_3_94 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_94 * 32) + (ax2_0_3_94 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_94 * 32) + (ax2_0_3_94 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_94 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_94 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_94 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_94 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_94 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_94 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_94 * 32) + (ax2_0_3_94 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_94 * 32) + (ax2_0_3_94 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_94 * 32) + (ax2_0_3_94 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_94 * 32) + (ax2_0_3_94 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_94 * 32) + (ax2_0_3_94 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_94 * 32) + (ax2_0_3_94 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_94 * 32) + (ax2_0_3_94 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_94 * 32) + (ax2_0_3_94 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_94 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_94 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_94 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_94 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_94 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_94 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_94 * 32) + (ax2_0_3_94 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_94 * 32) + (ax2_0_3_94 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_94 * 32) + (ax2_0_3_94 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_94 * 32) + (ax2_0_3_94 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_196 = 0; ax0_ax1_fused_0_196 < 4; ++ax0_ax1_fused_0_196) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_196 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_196 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3136))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_197 = 0; ax0_ax1_fused_0_197 < 4; ++ax0_ax1_fused_0_197) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_197 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_197 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3136))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_95 = 0; ax3_0_1_95 < 2; ++ax3_0_1_95) {
    for (int ax0_0_190 = 0; ax0_0_190 < 4; ++ax0_0_190) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_190 * 512)) + (ax3_0_1_95 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_190 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_190 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_190 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_190 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_191 = 0; ax0_0_191 < 4; ++ax0_0_191) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_191 * 512)) + (ax3_0_1_95 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_191 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_191 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_191 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_191 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_95 = 0; ax1_0_3_95 < 4; ++ax1_0_3_95) {
      for (int ax2_0_3_95 = 0; ax2_0_3_95 < 4; ++ax2_0_3_95) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_95 * 32) + (ax2_0_3_95 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_95 * 32) + (ax2_0_3_95 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_95 * 32) + (ax2_0_3_95 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_95 * 32) + (ax2_0_3_95 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_95 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_95 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_95 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_95 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_95 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_95 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_95 * 32) + (ax2_0_3_95 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_95 * 32) + (ax2_0_3_95 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_95 * 32) + (ax2_0_3_95 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_95 * 32) + (ax2_0_3_95 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_95 * 32) + (ax2_0_3_95 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_95 * 32) + (ax2_0_3_95 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_95 * 32) + (ax2_0_3_95 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_95 * 32) + (ax2_0_3_95 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_95 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_95 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_95 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_95 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_95 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_95 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_95 * 32) + (ax2_0_3_95 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_95 * 32) + (ax2_0_3_95 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_95 * 32) + (ax2_0_3_95 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_95 * 32) + (ax2_0_3_95 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_198 = 0; ax0_ax1_fused_0_198 < 4; ++ax0_ax1_fused_0_198) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_198 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_198 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3168))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_199 = 0; ax0_ax1_fused_0_199 < 4; ++ax0_ax1_fused_0_199) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_199 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_199 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3168))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_96 = 0; ax3_0_1_96 < 2; ++ax3_0_1_96) {
    for (int ax0_0_192 = 0; ax0_0_192 < 4; ++ax0_0_192) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_192 * 512)) + (ax3_0_1_96 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_192 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_192 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_192 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_192 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_193 = 0; ax0_0_193 < 4; ++ax0_0_193) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_193 * 512)) + (ax3_0_1_96 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_193 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_193 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_193 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_193 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_96 = 0; ax1_0_3_96 < 4; ++ax1_0_3_96) {
      for (int ax2_0_3_96 = 0; ax2_0_3_96 < 4; ++ax2_0_3_96) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_96 * 32) + (ax2_0_3_96 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_96 * 32) + (ax2_0_3_96 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_96 * 32) + (ax2_0_3_96 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_96 * 32) + (ax2_0_3_96 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_96 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_96 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_96 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_96 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_96 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_96 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_96 * 32) + (ax2_0_3_96 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_96 * 32) + (ax2_0_3_96 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_96 * 32) + (ax2_0_3_96 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_96 * 32) + (ax2_0_3_96 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_96 * 32) + (ax2_0_3_96 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_96 * 32) + (ax2_0_3_96 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_96 * 32) + (ax2_0_3_96 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_96 * 32) + (ax2_0_3_96 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_96 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_96 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_96 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_96 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_96 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_96 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_96 * 32) + (ax2_0_3_96 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_96 * 32) + (ax2_0_3_96 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_96 * 32) + (ax2_0_3_96 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_96 * 32) + (ax2_0_3_96 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_200 = 0; ax0_ax1_fused_0_200 < 4; ++ax0_ax1_fused_0_200) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_200 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_200 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3200))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_201 = 0; ax0_ax1_fused_0_201 < 4; ++ax0_ax1_fused_0_201) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_201 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_201 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3200))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_97 = 0; ax3_0_1_97 < 2; ++ax3_0_1_97) {
    for (int ax0_0_194 = 0; ax0_0_194 < 4; ++ax0_0_194) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_194 * 512)) + (ax3_0_1_97 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_194 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_194 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_194 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_194 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_195 = 0; ax0_0_195 < 4; ++ax0_0_195) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_195 * 512)) + (ax3_0_1_97 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_195 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_195 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_195 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_195 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_97 = 0; ax1_0_3_97 < 4; ++ax1_0_3_97) {
      for (int ax2_0_3_97 = 0; ax2_0_3_97 < 4; ++ax2_0_3_97) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_97 * 32) + (ax2_0_3_97 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_97 * 32) + (ax2_0_3_97 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_97 * 32) + (ax2_0_3_97 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_97 * 32) + (ax2_0_3_97 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_97 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_97 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_97 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_97 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_97 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_97 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_97 * 32) + (ax2_0_3_97 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_97 * 32) + (ax2_0_3_97 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_97 * 32) + (ax2_0_3_97 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_97 * 32) + (ax2_0_3_97 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_97 * 32) + (ax2_0_3_97 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_97 * 32) + (ax2_0_3_97 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_97 * 32) + (ax2_0_3_97 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_97 * 32) + (ax2_0_3_97 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_97 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_97 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_97 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_97 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_97 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_97 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_97 * 32) + (ax2_0_3_97 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_97 * 32) + (ax2_0_3_97 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_97 * 32) + (ax2_0_3_97 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_97 * 32) + (ax2_0_3_97 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_202 = 0; ax0_ax1_fused_0_202 < 4; ++ax0_ax1_fused_0_202) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_202 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_202 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3232))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_203 = 0; ax0_ax1_fused_0_203 < 4; ++ax0_ax1_fused_0_203) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_203 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_203 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3232))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_98 = 0; ax3_0_1_98 < 2; ++ax3_0_1_98) {
    for (int ax0_0_196 = 0; ax0_0_196 < 4; ++ax0_0_196) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_196 * 512)) + (ax3_0_1_98 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_196 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_196 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_196 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_196 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_197 = 0; ax0_0_197 < 4; ++ax0_0_197) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_197 * 512)) + (ax3_0_1_98 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_197 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_197 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_197 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_197 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_98 = 0; ax1_0_3_98 < 4; ++ax1_0_3_98) {
      for (int ax2_0_3_98 = 0; ax2_0_3_98 < 4; ++ax2_0_3_98) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_98 * 32) + (ax2_0_3_98 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_98 * 32) + (ax2_0_3_98 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_98 * 32) + (ax2_0_3_98 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_98 * 32) + (ax2_0_3_98 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_98 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_98 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_98 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_98 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_98 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_98 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_98 * 32) + (ax2_0_3_98 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_98 * 32) + (ax2_0_3_98 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_98 * 32) + (ax2_0_3_98 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_98 * 32) + (ax2_0_3_98 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_98 * 32) + (ax2_0_3_98 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_98 * 32) + (ax2_0_3_98 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_98 * 32) + (ax2_0_3_98 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_98 * 32) + (ax2_0_3_98 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_98 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_98 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_98 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_98 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_98 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_98 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_98 * 32) + (ax2_0_3_98 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_98 * 32) + (ax2_0_3_98 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_98 * 32) + (ax2_0_3_98 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_98 * 32) + (ax2_0_3_98 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_204 = 0; ax0_ax1_fused_0_204 < 4; ++ax0_ax1_fused_0_204) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_204 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_204 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3264))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_205 = 0; ax0_ax1_fused_0_205 < 4; ++ax0_ax1_fused_0_205) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_205 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_205 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3264))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_99 = 0; ax3_0_1_99 < 2; ++ax3_0_1_99) {
    for (int ax0_0_198 = 0; ax0_0_198 < 4; ++ax0_0_198) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_198 * 512)) + (ax3_0_1_99 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_198 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_198 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_198 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_198 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_199 = 0; ax0_0_199 < 4; ++ax0_0_199) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_199 * 512)) + (ax3_0_1_99 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_199 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_199 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_199 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_199 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_99 = 0; ax1_0_3_99 < 4; ++ax1_0_3_99) {
      for (int ax2_0_3_99 = 0; ax2_0_3_99 < 4; ++ax2_0_3_99) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_99 * 32) + (ax2_0_3_99 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_99 * 32) + (ax2_0_3_99 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_99 * 32) + (ax2_0_3_99 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_99 * 32) + (ax2_0_3_99 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_99 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_99 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_99 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_99 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_99 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_99 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_99 * 32) + (ax2_0_3_99 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_99 * 32) + (ax2_0_3_99 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_99 * 32) + (ax2_0_3_99 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_99 * 32) + (ax2_0_3_99 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_99 * 32) + (ax2_0_3_99 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_99 * 32) + (ax2_0_3_99 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_99 * 32) + (ax2_0_3_99 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_99 * 32) + (ax2_0_3_99 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_99 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_99 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_99 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_99 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_99 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_99 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_99 * 32) + (ax2_0_3_99 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_99 * 32) + (ax2_0_3_99 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_99 * 32) + (ax2_0_3_99 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_99 * 32) + (ax2_0_3_99 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_206 = 0; ax0_ax1_fused_0_206 < 4; ++ax0_ax1_fused_0_206) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_206 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_206 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3296))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_207 = 0; ax0_ax1_fused_0_207 < 4; ++ax0_ax1_fused_0_207) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_207 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_207 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3296))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_100 = 0; ax3_0_1_100 < 2; ++ax3_0_1_100) {
    for (int ax0_0_200 = 0; ax0_0_200 < 4; ++ax0_0_200) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_200 * 512)) + (ax3_0_1_100 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_200 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_200 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_200 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_200 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_201 = 0; ax0_0_201 < 4; ++ax0_0_201) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_201 * 512)) + (ax3_0_1_100 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_201 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_201 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_201 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_201 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_100 = 0; ax1_0_3_100 < 4; ++ax1_0_3_100) {
      for (int ax2_0_3_100 = 0; ax2_0_3_100 < 4; ++ax2_0_3_100) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_100 * 32) + (ax2_0_3_100 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_100 * 32) + (ax2_0_3_100 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_100 * 32) + (ax2_0_3_100 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_100 * 32) + (ax2_0_3_100 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_100 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_100 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_100 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_100 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_100 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_100 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_100 * 32) + (ax2_0_3_100 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_100 * 32) + (ax2_0_3_100 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_100 * 32) + (ax2_0_3_100 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_100 * 32) + (ax2_0_3_100 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_100 * 32) + (ax2_0_3_100 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_100 * 32) + (ax2_0_3_100 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_100 * 32) + (ax2_0_3_100 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_100 * 32) + (ax2_0_3_100 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_100 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_100 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_100 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_100 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_100 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_100 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_100 * 32) + (ax2_0_3_100 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_100 * 32) + (ax2_0_3_100 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_100 * 32) + (ax2_0_3_100 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_100 * 32) + (ax2_0_3_100 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_208 = 0; ax0_ax1_fused_0_208 < 4; ++ax0_ax1_fused_0_208) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_208 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_208 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3328))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_209 = 0; ax0_ax1_fused_0_209 < 4; ++ax0_ax1_fused_0_209) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_209 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_209 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3328))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_101 = 0; ax3_0_1_101 < 2; ++ax3_0_1_101) {
    for (int ax0_0_202 = 0; ax0_0_202 < 4; ++ax0_0_202) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_202 * 512)) + (ax3_0_1_101 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_202 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_202 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_202 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_202 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_203 = 0; ax0_0_203 < 4; ++ax0_0_203) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_203 * 512)) + (ax3_0_1_101 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_203 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_203 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_203 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_203 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_101 = 0; ax1_0_3_101 < 4; ++ax1_0_3_101) {
      for (int ax2_0_3_101 = 0; ax2_0_3_101 < 4; ++ax2_0_3_101) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_101 * 32) + (ax2_0_3_101 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_101 * 32) + (ax2_0_3_101 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_101 * 32) + (ax2_0_3_101 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_101 * 32) + (ax2_0_3_101 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_101 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_101 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_101 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_101 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_101 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_101 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_101 * 32) + (ax2_0_3_101 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_101 * 32) + (ax2_0_3_101 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_101 * 32) + (ax2_0_3_101 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_101 * 32) + (ax2_0_3_101 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_101 * 32) + (ax2_0_3_101 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_101 * 32) + (ax2_0_3_101 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_101 * 32) + (ax2_0_3_101 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_101 * 32) + (ax2_0_3_101 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_101 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_101 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_101 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_101 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_101 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_101 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_101 * 32) + (ax2_0_3_101 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_101 * 32) + (ax2_0_3_101 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_101 * 32) + (ax2_0_3_101 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_101 * 32) + (ax2_0_3_101 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_210 = 0; ax0_ax1_fused_0_210 < 4; ++ax0_ax1_fused_0_210) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_210 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_210 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3360))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_211 = 0; ax0_ax1_fused_0_211 < 4; ++ax0_ax1_fused_0_211) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_211 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_211 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3360))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_102 = 0; ax3_0_1_102 < 2; ++ax3_0_1_102) {
    for (int ax0_0_204 = 0; ax0_0_204 < 4; ++ax0_0_204) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_204 * 512)) + (ax3_0_1_102 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_204 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_204 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_204 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_204 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_205 = 0; ax0_0_205 < 4; ++ax0_0_205) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_205 * 512)) + (ax3_0_1_102 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_205 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_205 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_205 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_205 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_102 = 0; ax1_0_3_102 < 4; ++ax1_0_3_102) {
      for (int ax2_0_3_102 = 0; ax2_0_3_102 < 4; ++ax2_0_3_102) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_102 * 32) + (ax2_0_3_102 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_102 * 32) + (ax2_0_3_102 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_102 * 32) + (ax2_0_3_102 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_102 * 32) + (ax2_0_3_102 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_102 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_102 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_102 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_102 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_102 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_102 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_102 * 32) + (ax2_0_3_102 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_102 * 32) + (ax2_0_3_102 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_102 * 32) + (ax2_0_3_102 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_102 * 32) + (ax2_0_3_102 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_102 * 32) + (ax2_0_3_102 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_102 * 32) + (ax2_0_3_102 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_102 * 32) + (ax2_0_3_102 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_102 * 32) + (ax2_0_3_102 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_102 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_102 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_102 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_102 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_102 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_102 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_102 * 32) + (ax2_0_3_102 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_102 * 32) + (ax2_0_3_102 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_102 * 32) + (ax2_0_3_102 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_102 * 32) + (ax2_0_3_102 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_212 = 0; ax0_ax1_fused_0_212 < 4; ++ax0_ax1_fused_0_212) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_212 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_212 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3392))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_213 = 0; ax0_ax1_fused_0_213 < 4; ++ax0_ax1_fused_0_213) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_213 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_213 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3392))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_103 = 0; ax3_0_1_103 < 2; ++ax3_0_1_103) {
    for (int ax0_0_206 = 0; ax0_0_206 < 4; ++ax0_0_206) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_206 * 512)) + (ax3_0_1_103 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_206 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_206 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_206 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_206 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_207 = 0; ax0_0_207 < 4; ++ax0_0_207) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_207 * 512)) + (ax3_0_1_103 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_207 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_207 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_207 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_207 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_103 = 0; ax1_0_3_103 < 4; ++ax1_0_3_103) {
      for (int ax2_0_3_103 = 0; ax2_0_3_103 < 4; ++ax2_0_3_103) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_103 * 32) + (ax2_0_3_103 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_103 * 32) + (ax2_0_3_103 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_103 * 32) + (ax2_0_3_103 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_103 * 32) + (ax2_0_3_103 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_103 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_103 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_103 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_103 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_103 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_103 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_103 * 32) + (ax2_0_3_103 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_103 * 32) + (ax2_0_3_103 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_103 * 32) + (ax2_0_3_103 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_103 * 32) + (ax2_0_3_103 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_103 * 32) + (ax2_0_3_103 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_103 * 32) + (ax2_0_3_103 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_103 * 32) + (ax2_0_3_103 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_103 * 32) + (ax2_0_3_103 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_103 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_103 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_103 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_103 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_103 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_103 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_103 * 32) + (ax2_0_3_103 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_103 * 32) + (ax2_0_3_103 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_103 * 32) + (ax2_0_3_103 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_103 * 32) + (ax2_0_3_103 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_214 = 0; ax0_ax1_fused_0_214 < 4; ++ax0_ax1_fused_0_214) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_214 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_214 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3424))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_215 = 0; ax0_ax1_fused_0_215 < 4; ++ax0_ax1_fused_0_215) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_215 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_215 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3424))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_104 = 0; ax3_0_1_104 < 2; ++ax3_0_1_104) {
    for (int ax0_0_208 = 0; ax0_0_208 < 4; ++ax0_0_208) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_208 * 512)) + (ax3_0_1_104 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_208 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_208 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_208 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_208 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_209 = 0; ax0_0_209 < 4; ++ax0_0_209) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_209 * 512)) + (ax3_0_1_104 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_209 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_209 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_209 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_209 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_104 = 0; ax1_0_3_104 < 4; ++ax1_0_3_104) {
      for (int ax2_0_3_104 = 0; ax2_0_3_104 < 4; ++ax2_0_3_104) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_104 * 32) + (ax2_0_3_104 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_104 * 32) + (ax2_0_3_104 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_104 * 32) + (ax2_0_3_104 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_104 * 32) + (ax2_0_3_104 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_104 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_104 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_104 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_104 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_104 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_104 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_104 * 32) + (ax2_0_3_104 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_104 * 32) + (ax2_0_3_104 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_104 * 32) + (ax2_0_3_104 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_104 * 32) + (ax2_0_3_104 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_104 * 32) + (ax2_0_3_104 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_104 * 32) + (ax2_0_3_104 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_104 * 32) + (ax2_0_3_104 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_104 * 32) + (ax2_0_3_104 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_104 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_104 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_104 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_104 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_104 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_104 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_104 * 32) + (ax2_0_3_104 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_104 * 32) + (ax2_0_3_104 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_104 * 32) + (ax2_0_3_104 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_104 * 32) + (ax2_0_3_104 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_216 = 0; ax0_ax1_fused_0_216 < 4; ++ax0_ax1_fused_0_216) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_216 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_216 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3456))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_217 = 0; ax0_ax1_fused_0_217 < 4; ++ax0_ax1_fused_0_217) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_217 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_217 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3456))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_105 = 0; ax3_0_1_105 < 2; ++ax3_0_1_105) {
    for (int ax0_0_210 = 0; ax0_0_210 < 4; ++ax0_0_210) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_210 * 512)) + (ax3_0_1_105 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_210 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_210 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_210 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_210 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_211 = 0; ax0_0_211 < 4; ++ax0_0_211) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_211 * 512)) + (ax3_0_1_105 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_211 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_211 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_211 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_211 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_105 = 0; ax1_0_3_105 < 4; ++ax1_0_3_105) {
      for (int ax2_0_3_105 = 0; ax2_0_3_105 < 4; ++ax2_0_3_105) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_105 * 32) + (ax2_0_3_105 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_105 * 32) + (ax2_0_3_105 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_105 * 32) + (ax2_0_3_105 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_105 * 32) + (ax2_0_3_105 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_105 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_105 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_105 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_105 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_105 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_105 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_105 * 32) + (ax2_0_3_105 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_105 * 32) + (ax2_0_3_105 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_105 * 32) + (ax2_0_3_105 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_105 * 32) + (ax2_0_3_105 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_105 * 32) + (ax2_0_3_105 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_105 * 32) + (ax2_0_3_105 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_105 * 32) + (ax2_0_3_105 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_105 * 32) + (ax2_0_3_105 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_105 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_105 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_105 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_105 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_105 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_105 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_105 * 32) + (ax2_0_3_105 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_105 * 32) + (ax2_0_3_105 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_105 * 32) + (ax2_0_3_105 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_105 * 32) + (ax2_0_3_105 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_218 = 0; ax0_ax1_fused_0_218 < 4; ++ax0_ax1_fused_0_218) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_218 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_218 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3488))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_219 = 0; ax0_ax1_fused_0_219 < 4; ++ax0_ax1_fused_0_219) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_219 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_219 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3488))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_106 = 0; ax3_0_1_106 < 2; ++ax3_0_1_106) {
    for (int ax0_0_212 = 0; ax0_0_212 < 4; ++ax0_0_212) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_212 * 512)) + (ax3_0_1_106 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_212 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_212 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_212 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_212 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_213 = 0; ax0_0_213 < 4; ++ax0_0_213) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_213 * 512)) + (ax3_0_1_106 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_213 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_213 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_213 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_213 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_106 = 0; ax1_0_3_106 < 4; ++ax1_0_3_106) {
      for (int ax2_0_3_106 = 0; ax2_0_3_106 < 4; ++ax2_0_3_106) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_106 * 32) + (ax2_0_3_106 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_106 * 32) + (ax2_0_3_106 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_106 * 32) + (ax2_0_3_106 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_106 * 32) + (ax2_0_3_106 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_106 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_106 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_106 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_106 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_106 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_106 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_106 * 32) + (ax2_0_3_106 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_106 * 32) + (ax2_0_3_106 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_106 * 32) + (ax2_0_3_106 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_106 * 32) + (ax2_0_3_106 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_106 * 32) + (ax2_0_3_106 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_106 * 32) + (ax2_0_3_106 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_106 * 32) + (ax2_0_3_106 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_106 * 32) + (ax2_0_3_106 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_106 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_106 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_106 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_106 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_106 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_106 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_106 * 32) + (ax2_0_3_106 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_106 * 32) + (ax2_0_3_106 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_106 * 32) + (ax2_0_3_106 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_106 * 32) + (ax2_0_3_106 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_220 = 0; ax0_ax1_fused_0_220 < 4; ++ax0_ax1_fused_0_220) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_220 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_220 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3520))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_221 = 0; ax0_ax1_fused_0_221 < 4; ++ax0_ax1_fused_0_221) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_221 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_221 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3520))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_107 = 0; ax3_0_1_107 < 2; ++ax3_0_1_107) {
    for (int ax0_0_214 = 0; ax0_0_214 < 4; ++ax0_0_214) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_214 * 512)) + (ax3_0_1_107 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_214 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_214 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_214 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_214 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_215 = 0; ax0_0_215 < 4; ++ax0_0_215) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_215 * 512)) + (ax3_0_1_107 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_215 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_215 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_215 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_215 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_107 = 0; ax1_0_3_107 < 4; ++ax1_0_3_107) {
      for (int ax2_0_3_107 = 0; ax2_0_3_107 < 4; ++ax2_0_3_107) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_107 * 32) + (ax2_0_3_107 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_107 * 32) + (ax2_0_3_107 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_107 * 32) + (ax2_0_3_107 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_107 * 32) + (ax2_0_3_107 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_107 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_107 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_107 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_107 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_107 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_107 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_107 * 32) + (ax2_0_3_107 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_107 * 32) + (ax2_0_3_107 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_107 * 32) + (ax2_0_3_107 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_107 * 32) + (ax2_0_3_107 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_107 * 32) + (ax2_0_3_107 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_107 * 32) + (ax2_0_3_107 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_107 * 32) + (ax2_0_3_107 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_107 * 32) + (ax2_0_3_107 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_107 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_107 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_107 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_107 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_107 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_107 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_107 * 32) + (ax2_0_3_107 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_107 * 32) + (ax2_0_3_107 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_107 * 32) + (ax2_0_3_107 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_107 * 32) + (ax2_0_3_107 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_222 = 0; ax0_ax1_fused_0_222 < 4; ++ax0_ax1_fused_0_222) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_222 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_222 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3552))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_223 = 0; ax0_ax1_fused_0_223 < 4; ++ax0_ax1_fused_0_223) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_223 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_223 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3552))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_108 = 0; ax3_0_1_108 < 2; ++ax3_0_1_108) {
    for (int ax0_0_216 = 0; ax0_0_216 < 4; ++ax0_0_216) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_216 * 512)) + (ax3_0_1_108 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_216 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_216 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_216 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_216 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_217 = 0; ax0_0_217 < 4; ++ax0_0_217) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_217 * 512)) + (ax3_0_1_108 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_217 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_217 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_217 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_217 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_108 = 0; ax1_0_3_108 < 4; ++ax1_0_3_108) {
      for (int ax2_0_3_108 = 0; ax2_0_3_108 < 4; ++ax2_0_3_108) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_108 * 32) + (ax2_0_3_108 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_108 * 32) + (ax2_0_3_108 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_108 * 32) + (ax2_0_3_108 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_108 * 32) + (ax2_0_3_108 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_108 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_108 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_108 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_108 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_108 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_108 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_108 * 32) + (ax2_0_3_108 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_108 * 32) + (ax2_0_3_108 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_108 * 32) + (ax2_0_3_108 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_108 * 32) + (ax2_0_3_108 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_108 * 32) + (ax2_0_3_108 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_108 * 32) + (ax2_0_3_108 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_108 * 32) + (ax2_0_3_108 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_108 * 32) + (ax2_0_3_108 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_108 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_108 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_108 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_108 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_108 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_108 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_108 * 32) + (ax2_0_3_108 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_108 * 32) + (ax2_0_3_108 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_108 * 32) + (ax2_0_3_108 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_108 * 32) + (ax2_0_3_108 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_224 = 0; ax0_ax1_fused_0_224 < 4; ++ax0_ax1_fused_0_224) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_224 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_224 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3584))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_225 = 0; ax0_ax1_fused_0_225 < 4; ++ax0_ax1_fused_0_225) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_225 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_225 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3584))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_109 = 0; ax3_0_1_109 < 2; ++ax3_0_1_109) {
    for (int ax0_0_218 = 0; ax0_0_218 < 4; ++ax0_0_218) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_218 * 512)) + (ax3_0_1_109 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_218 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_218 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_218 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_218 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_219 = 0; ax0_0_219 < 4; ++ax0_0_219) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_219 * 512)) + (ax3_0_1_109 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_219 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_219 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_219 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_219 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_109 = 0; ax1_0_3_109 < 4; ++ax1_0_3_109) {
      for (int ax2_0_3_109 = 0; ax2_0_3_109 < 4; ++ax2_0_3_109) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_109 * 32) + (ax2_0_3_109 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_109 * 32) + (ax2_0_3_109 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_109 * 32) + (ax2_0_3_109 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_109 * 32) + (ax2_0_3_109 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_109 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_109 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_109 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_109 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_109 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_109 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_109 * 32) + (ax2_0_3_109 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_109 * 32) + (ax2_0_3_109 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_109 * 32) + (ax2_0_3_109 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_109 * 32) + (ax2_0_3_109 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_109 * 32) + (ax2_0_3_109 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_109 * 32) + (ax2_0_3_109 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_109 * 32) + (ax2_0_3_109 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_109 * 32) + (ax2_0_3_109 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_109 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_109 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_109 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_109 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_109 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_109 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_109 * 32) + (ax2_0_3_109 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_109 * 32) + (ax2_0_3_109 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_109 * 32) + (ax2_0_3_109 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_109 * 32) + (ax2_0_3_109 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_226 = 0; ax0_ax1_fused_0_226 < 4; ++ax0_ax1_fused_0_226) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_226 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_226 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3616))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_227 = 0; ax0_ax1_fused_0_227 < 4; ++ax0_ax1_fused_0_227) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_227 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_227 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3616))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_110 = 0; ax3_0_1_110 < 2; ++ax3_0_1_110) {
    for (int ax0_0_220 = 0; ax0_0_220 < 4; ++ax0_0_220) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_220 * 512)) + (ax3_0_1_110 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_220 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_220 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_220 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_220 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_221 = 0; ax0_0_221 < 4; ++ax0_0_221) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_221 * 512)) + (ax3_0_1_110 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_221 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_221 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_221 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_221 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_110 = 0; ax1_0_3_110 < 4; ++ax1_0_3_110) {
      for (int ax2_0_3_110 = 0; ax2_0_3_110 < 4; ++ax2_0_3_110) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_110 * 32) + (ax2_0_3_110 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_110 * 32) + (ax2_0_3_110 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_110 * 32) + (ax2_0_3_110 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_110 * 32) + (ax2_0_3_110 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_110 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_110 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_110 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_110 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_110 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_110 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_110 * 32) + (ax2_0_3_110 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_110 * 32) + (ax2_0_3_110 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_110 * 32) + (ax2_0_3_110 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_110 * 32) + (ax2_0_3_110 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_110 * 32) + (ax2_0_3_110 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_110 * 32) + (ax2_0_3_110 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_110 * 32) + (ax2_0_3_110 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_110 * 32) + (ax2_0_3_110 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_110 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_110 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_110 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_110 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_110 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_110 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_110 * 32) + (ax2_0_3_110 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_110 * 32) + (ax2_0_3_110 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_110 * 32) + (ax2_0_3_110 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_110 * 32) + (ax2_0_3_110 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_228 = 0; ax0_ax1_fused_0_228 < 4; ++ax0_ax1_fused_0_228) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_228 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_228 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3648))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_229 = 0; ax0_ax1_fused_0_229 < 4; ++ax0_ax1_fused_0_229) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_229 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_229 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3648))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_111 = 0; ax3_0_1_111 < 2; ++ax3_0_1_111) {
    for (int ax0_0_222 = 0; ax0_0_222 < 4; ++ax0_0_222) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_222 * 512)) + (ax3_0_1_111 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_222 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_222 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_222 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_222 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_223 = 0; ax0_0_223 < 4; ++ax0_0_223) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_223 * 512)) + (ax3_0_1_111 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_223 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_223 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_223 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_223 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_111 = 0; ax1_0_3_111 < 4; ++ax1_0_3_111) {
      for (int ax2_0_3_111 = 0; ax2_0_3_111 < 4; ++ax2_0_3_111) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_111 * 32) + (ax2_0_3_111 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_111 * 32) + (ax2_0_3_111 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_111 * 32) + (ax2_0_3_111 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_111 * 32) + (ax2_0_3_111 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_111 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_111 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_111 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_111 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_111 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_111 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_111 * 32) + (ax2_0_3_111 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_111 * 32) + (ax2_0_3_111 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_111 * 32) + (ax2_0_3_111 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_111 * 32) + (ax2_0_3_111 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_111 * 32) + (ax2_0_3_111 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_111 * 32) + (ax2_0_3_111 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_111 * 32) + (ax2_0_3_111 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_111 * 32) + (ax2_0_3_111 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_111 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_111 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_111 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_111 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_111 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_111 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_111 * 32) + (ax2_0_3_111 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_111 * 32) + (ax2_0_3_111 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_111 * 32) + (ax2_0_3_111 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_111 * 32) + (ax2_0_3_111 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_230 = 0; ax0_ax1_fused_0_230 < 4; ++ax0_ax1_fused_0_230) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_230 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_230 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3680))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_231 = 0; ax0_ax1_fused_0_231 < 4; ++ax0_ax1_fused_0_231) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_231 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_231 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3680))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_112 = 0; ax3_0_1_112 < 2; ++ax3_0_1_112) {
    for (int ax0_0_224 = 0; ax0_0_224 < 4; ++ax0_0_224) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_224 * 512)) + (ax3_0_1_112 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_224 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_224 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_224 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_224 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_225 = 0; ax0_0_225 < 4; ++ax0_0_225) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_225 * 512)) + (ax3_0_1_112 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_225 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_225 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_225 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_225 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_112 = 0; ax1_0_3_112 < 4; ++ax1_0_3_112) {
      for (int ax2_0_3_112 = 0; ax2_0_3_112 < 4; ++ax2_0_3_112) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_112 * 32) + (ax2_0_3_112 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_112 * 32) + (ax2_0_3_112 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_112 * 32) + (ax2_0_3_112 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_112 * 32) + (ax2_0_3_112 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_112 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_112 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_112 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_112 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_112 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_112 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_112 * 32) + (ax2_0_3_112 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_112 * 32) + (ax2_0_3_112 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_112 * 32) + (ax2_0_3_112 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_112 * 32) + (ax2_0_3_112 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_112 * 32) + (ax2_0_3_112 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_112 * 32) + (ax2_0_3_112 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_112 * 32) + (ax2_0_3_112 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_112 * 32) + (ax2_0_3_112 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_112 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_112 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_112 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_112 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_112 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_112 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_112 * 32) + (ax2_0_3_112 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_112 * 32) + (ax2_0_3_112 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_112 * 32) + (ax2_0_3_112 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_112 * 32) + (ax2_0_3_112 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_232 = 0; ax0_ax1_fused_0_232 < 4; ++ax0_ax1_fused_0_232) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_232 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_232 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3712))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_233 = 0; ax0_ax1_fused_0_233 < 4; ++ax0_ax1_fused_0_233) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_233 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_233 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3712))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_113 = 0; ax3_0_1_113 < 2; ++ax3_0_1_113) {
    for (int ax0_0_226 = 0; ax0_0_226 < 4; ++ax0_0_226) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_226 * 512)) + (ax3_0_1_113 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_226 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_226 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_226 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_226 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_227 = 0; ax0_0_227 < 4; ++ax0_0_227) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_227 * 512)) + (ax3_0_1_113 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_227 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_227 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_227 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_227 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_113 = 0; ax1_0_3_113 < 4; ++ax1_0_3_113) {
      for (int ax2_0_3_113 = 0; ax2_0_3_113 < 4; ++ax2_0_3_113) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_113 * 32) + (ax2_0_3_113 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_113 * 32) + (ax2_0_3_113 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_113 * 32) + (ax2_0_3_113 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_113 * 32) + (ax2_0_3_113 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_113 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_113 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_113 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_113 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_113 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_113 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_113 * 32) + (ax2_0_3_113 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_113 * 32) + (ax2_0_3_113 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_113 * 32) + (ax2_0_3_113 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_113 * 32) + (ax2_0_3_113 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_113 * 32) + (ax2_0_3_113 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_113 * 32) + (ax2_0_3_113 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_113 * 32) + (ax2_0_3_113 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_113 * 32) + (ax2_0_3_113 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_113 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_113 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_113 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_113 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_113 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_113 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_113 * 32) + (ax2_0_3_113 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_113 * 32) + (ax2_0_3_113 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_113 * 32) + (ax2_0_3_113 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_113 * 32) + (ax2_0_3_113 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_234 = 0; ax0_ax1_fused_0_234 < 4; ++ax0_ax1_fused_0_234) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_234 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_234 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3744))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_235 = 0; ax0_ax1_fused_0_235 < 4; ++ax0_ax1_fused_0_235) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_235 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_235 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3744))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_114 = 0; ax3_0_1_114 < 2; ++ax3_0_1_114) {
    for (int ax0_0_228 = 0; ax0_0_228 < 4; ++ax0_0_228) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_228 * 512)) + (ax3_0_1_114 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_228 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_228 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_228 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_228 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_229 = 0; ax0_0_229 < 4; ++ax0_0_229) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_229 * 512)) + (ax3_0_1_114 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_229 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_229 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_229 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_229 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_114 = 0; ax1_0_3_114 < 4; ++ax1_0_3_114) {
      for (int ax2_0_3_114 = 0; ax2_0_3_114 < 4; ++ax2_0_3_114) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_114 * 32) + (ax2_0_3_114 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_114 * 32) + (ax2_0_3_114 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_114 * 32) + (ax2_0_3_114 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_114 * 32) + (ax2_0_3_114 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_114 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_114 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_114 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_114 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_114 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_114 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_114 * 32) + (ax2_0_3_114 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_114 * 32) + (ax2_0_3_114 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_114 * 32) + (ax2_0_3_114 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_114 * 32) + (ax2_0_3_114 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_114 * 32) + (ax2_0_3_114 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_114 * 32) + (ax2_0_3_114 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_114 * 32) + (ax2_0_3_114 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_114 * 32) + (ax2_0_3_114 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_114 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_114 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_114 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_114 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_114 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_114 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_114 * 32) + (ax2_0_3_114 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_114 * 32) + (ax2_0_3_114 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_114 * 32) + (ax2_0_3_114 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_114 * 32) + (ax2_0_3_114 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_236 = 0; ax0_ax1_fused_0_236 < 4; ++ax0_ax1_fused_0_236) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_236 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_236 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3776))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_237 = 0; ax0_ax1_fused_0_237 < 4; ++ax0_ax1_fused_0_237) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_237 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_237 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3776))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_115 = 0; ax3_0_1_115 < 2; ++ax3_0_1_115) {
    for (int ax0_0_230 = 0; ax0_0_230 < 4; ++ax0_0_230) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_230 * 512)) + (ax3_0_1_115 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_230 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_230 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_230 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_230 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_231 = 0; ax0_0_231 < 4; ++ax0_0_231) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_231 * 512)) + (ax3_0_1_115 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_231 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_231 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_231 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_231 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_115 = 0; ax1_0_3_115 < 4; ++ax1_0_3_115) {
      for (int ax2_0_3_115 = 0; ax2_0_3_115 < 4; ++ax2_0_3_115) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_115 * 32) + (ax2_0_3_115 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_115 * 32) + (ax2_0_3_115 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_115 * 32) + (ax2_0_3_115 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_115 * 32) + (ax2_0_3_115 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_115 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_115 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_115 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_115 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_115 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_115 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_115 * 32) + (ax2_0_3_115 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_115 * 32) + (ax2_0_3_115 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_115 * 32) + (ax2_0_3_115 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_115 * 32) + (ax2_0_3_115 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_115 * 32) + (ax2_0_3_115 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_115 * 32) + (ax2_0_3_115 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_115 * 32) + (ax2_0_3_115 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_115 * 32) + (ax2_0_3_115 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_115 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_115 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_115 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_115 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_115 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_115 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_115 * 32) + (ax2_0_3_115 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_115 * 32) + (ax2_0_3_115 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_115 * 32) + (ax2_0_3_115 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_115 * 32) + (ax2_0_3_115 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_238 = 0; ax0_ax1_fused_0_238 < 4; ++ax0_ax1_fused_0_238) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_238 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_238 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3808))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_239 = 0; ax0_ax1_fused_0_239 < 4; ++ax0_ax1_fused_0_239) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_239 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_239 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3808))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_116 = 0; ax3_0_1_116 < 2; ++ax3_0_1_116) {
    for (int ax0_0_232 = 0; ax0_0_232 < 4; ++ax0_0_232) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_232 * 512)) + (ax3_0_1_116 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_232 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_232 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_232 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_232 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_233 = 0; ax0_0_233 < 4; ++ax0_0_233) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_233 * 512)) + (ax3_0_1_116 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_233 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_233 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_233 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_233 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_116 = 0; ax1_0_3_116 < 4; ++ax1_0_3_116) {
      for (int ax2_0_3_116 = 0; ax2_0_3_116 < 4; ++ax2_0_3_116) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_116 * 32) + (ax2_0_3_116 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_116 * 32) + (ax2_0_3_116 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_116 * 32) + (ax2_0_3_116 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_116 * 32) + (ax2_0_3_116 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_116 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_116 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_116 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_116 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_116 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_116 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_116 * 32) + (ax2_0_3_116 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_116 * 32) + (ax2_0_3_116 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_116 * 32) + (ax2_0_3_116 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_116 * 32) + (ax2_0_3_116 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_116 * 32) + (ax2_0_3_116 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_116 * 32) + (ax2_0_3_116 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_116 * 32) + (ax2_0_3_116 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_116 * 32) + (ax2_0_3_116 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_116 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_116 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_116 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_116 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_116 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_116 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_116 * 32) + (ax2_0_3_116 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_116 * 32) + (ax2_0_3_116 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_116 * 32) + (ax2_0_3_116 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_116 * 32) + (ax2_0_3_116 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_240 = 0; ax0_ax1_fused_0_240 < 4; ++ax0_ax1_fused_0_240) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_240 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_240 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3840))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_241 = 0; ax0_ax1_fused_0_241 < 4; ++ax0_ax1_fused_0_241) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_241 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_241 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3840))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_117 = 0; ax3_0_1_117 < 2; ++ax3_0_1_117) {
    for (int ax0_0_234 = 0; ax0_0_234 < 4; ++ax0_0_234) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_234 * 512)) + (ax3_0_1_117 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_234 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_234 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_234 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_234 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_235 = 0; ax0_0_235 < 4; ++ax0_0_235) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_235 * 512)) + (ax3_0_1_117 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_235 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_235 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_235 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_235 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_117 = 0; ax1_0_3_117 < 4; ++ax1_0_3_117) {
      for (int ax2_0_3_117 = 0; ax2_0_3_117 < 4; ++ax2_0_3_117) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_117 * 32) + (ax2_0_3_117 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_117 * 32) + (ax2_0_3_117 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_117 * 32) + (ax2_0_3_117 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_117 * 32) + (ax2_0_3_117 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_117 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_117 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_117 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_117 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_117 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_117 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_117 * 32) + (ax2_0_3_117 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_117 * 32) + (ax2_0_3_117 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_117 * 32) + (ax2_0_3_117 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_117 * 32) + (ax2_0_3_117 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_117 * 32) + (ax2_0_3_117 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_117 * 32) + (ax2_0_3_117 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_117 * 32) + (ax2_0_3_117 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_117 * 32) + (ax2_0_3_117 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_117 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_117 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_117 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_117 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_117 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_117 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_117 * 32) + (ax2_0_3_117 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_117 * 32) + (ax2_0_3_117 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_117 * 32) + (ax2_0_3_117 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_117 * 32) + (ax2_0_3_117 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_242 = 0; ax0_ax1_fused_0_242 < 4; ++ax0_ax1_fused_0_242) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_242 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_242 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3872))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_243 = 0; ax0_ax1_fused_0_243 < 4; ++ax0_ax1_fused_0_243) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_243 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_243 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3872))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_118 = 0; ax3_0_1_118 < 2; ++ax3_0_1_118) {
    for (int ax0_0_236 = 0; ax0_0_236 < 4; ++ax0_0_236) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_236 * 512)) + (ax3_0_1_118 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_236 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_236 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_236 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_236 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_237 = 0; ax0_0_237 < 4; ++ax0_0_237) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_237 * 512)) + (ax3_0_1_118 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_237 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_237 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_237 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_237 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_118 = 0; ax1_0_3_118 < 4; ++ax1_0_3_118) {
      for (int ax2_0_3_118 = 0; ax2_0_3_118 < 4; ++ax2_0_3_118) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_118 * 32) + (ax2_0_3_118 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_118 * 32) + (ax2_0_3_118 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_118 * 32) + (ax2_0_3_118 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_118 * 32) + (ax2_0_3_118 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_118 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_118 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_118 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_118 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_118 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_118 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_118 * 32) + (ax2_0_3_118 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_118 * 32) + (ax2_0_3_118 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_118 * 32) + (ax2_0_3_118 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_118 * 32) + (ax2_0_3_118 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_118 * 32) + (ax2_0_3_118 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_118 * 32) + (ax2_0_3_118 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_118 * 32) + (ax2_0_3_118 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_118 * 32) + (ax2_0_3_118 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_118 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_118 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_118 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_118 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_118 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_118 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_118 * 32) + (ax2_0_3_118 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_118 * 32) + (ax2_0_3_118 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_118 * 32) + (ax2_0_3_118 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_118 * 32) + (ax2_0_3_118 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_244 = 0; ax0_ax1_fused_0_244 < 4; ++ax0_ax1_fused_0_244) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_244 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_244 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3904))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_245 = 0; ax0_ax1_fused_0_245 < 4; ++ax0_ax1_fused_0_245) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_245 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_245 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3904))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_119 = 0; ax3_0_1_119 < 2; ++ax3_0_1_119) {
    for (int ax0_0_238 = 0; ax0_0_238 < 4; ++ax0_0_238) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_238 * 512)) + (ax3_0_1_119 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_238 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_238 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_238 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_238 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_239 = 0; ax0_0_239 < 4; ++ax0_0_239) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_239 * 512)) + (ax3_0_1_119 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_239 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_239 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_239 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_239 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_119 = 0; ax1_0_3_119 < 4; ++ax1_0_3_119) {
      for (int ax2_0_3_119 = 0; ax2_0_3_119 < 4; ++ax2_0_3_119) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_119 * 32) + (ax2_0_3_119 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_119 * 32) + (ax2_0_3_119 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_119 * 32) + (ax2_0_3_119 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_119 * 32) + (ax2_0_3_119 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_119 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_119 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_119 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_119 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_119 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_119 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_119 * 32) + (ax2_0_3_119 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_119 * 32) + (ax2_0_3_119 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_119 * 32) + (ax2_0_3_119 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_119 * 32) + (ax2_0_3_119 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_119 * 32) + (ax2_0_3_119 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_119 * 32) + (ax2_0_3_119 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_119 * 32) + (ax2_0_3_119 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_119 * 32) + (ax2_0_3_119 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_119 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_119 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_119 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_119 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_119 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_119 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_119 * 32) + (ax2_0_3_119 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_119 * 32) + (ax2_0_3_119 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_119 * 32) + (ax2_0_3_119 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_119 * 32) + (ax2_0_3_119 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_246 = 0; ax0_ax1_fused_0_246 < 4; ++ax0_ax1_fused_0_246) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_246 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_246 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3936))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_247 = 0; ax0_ax1_fused_0_247 < 4; ++ax0_ax1_fused_0_247) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_247 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_247 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3936))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_120 = 0; ax3_0_1_120 < 2; ++ax3_0_1_120) {
    for (int ax0_0_240 = 0; ax0_0_240 < 4; ++ax0_0_240) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_240 * 512)) + (ax3_0_1_120 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_240 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_240 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_240 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_240 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_241 = 0; ax0_0_241 < 4; ++ax0_0_241) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_241 * 512)) + (ax3_0_1_120 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_241 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_241 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_241 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_241 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_120 = 0; ax1_0_3_120 < 4; ++ax1_0_3_120) {
      for (int ax2_0_3_120 = 0; ax2_0_3_120 < 4; ++ax2_0_3_120) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_120 * 32) + (ax2_0_3_120 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_120 * 32) + (ax2_0_3_120 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_120 * 32) + (ax2_0_3_120 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_120 * 32) + (ax2_0_3_120 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_120 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_120 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_120 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_120 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_120 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_120 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_120 * 32) + (ax2_0_3_120 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_120 * 32) + (ax2_0_3_120 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_120 * 32) + (ax2_0_3_120 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_120 * 32) + (ax2_0_3_120 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_120 * 32) + (ax2_0_3_120 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_120 * 32) + (ax2_0_3_120 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_120 * 32) + (ax2_0_3_120 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_120 * 32) + (ax2_0_3_120 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_120 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_120 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_120 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_120 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_120 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_120 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_120 * 32) + (ax2_0_3_120 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_120 * 32) + (ax2_0_3_120 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_120 * 32) + (ax2_0_3_120 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_120 * 32) + (ax2_0_3_120 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_248 = 0; ax0_ax1_fused_0_248 < 4; ++ax0_ax1_fused_0_248) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_248 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_248 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3968))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_249 = 0; ax0_ax1_fused_0_249 < 4; ++ax0_ax1_fused_0_249) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0_249 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_249 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 3968))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_121 = 0; ax3_0_1_121 < 2; ++ax3_0_1_121) {
    for (int ax0_0_242 = 0; ax0_0_242 < 4; ++ax0_0_242) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_242 * 512)) + (ax3_0_1_121 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_242 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_242 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_242 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_242 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_243 = 0; ax0_0_243 < 4; ++ax0_0_243) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_243 * 512)) + (ax3_0_1_121 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_243 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_243 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_243 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_243 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_121 = 0; ax1_0_3_121 < 4; ++ax1_0_3_121) {
      for (int ax2_0_3_121 = 0; ax2_0_3_121 < 4; ++ax2_0_3_121) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_121 * 32) + (ax2_0_3_121 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_121 * 32) + (ax2_0_3_121 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_121 * 32) + (ax2_0_3_121 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_121 * 32) + (ax2_0_3_121 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_121 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_121 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_121 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_121 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_121 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_121 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_121 * 32) + (ax2_0_3_121 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_121 * 32) + (ax2_0_3_121 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_121 * 32) + (ax2_0_3_121 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_121 * 32) + (ax2_0_3_121 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_121 * 32) + (ax2_0_3_121 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_121 * 32) + (ax2_0_3_121 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_121 * 32) + (ax2_0_3_121 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_121 * 32) + (ax2_0_3_121 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_121 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_121 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_121 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_121 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_121 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_121 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_121 * 32) + (ax2_0_3_121 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_121 * 32) + (ax2_0_3_121 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_121 * 32) + (ax2_0_3_121 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_121 * 32) + (ax2_0_3_121 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_250 = 0; ax0_ax1_fused_0_250 < 4; ++ax0_ax1_fused_0_250) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_250 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_250 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 4000))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_251 = 0; ax0_ax1_fused_0_251 < 4; ++ax0_ax1_fused_0_251) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_251 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_251 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 4000))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_122 = 0; ax3_0_1_122 < 2; ++ax3_0_1_122) {
    for (int ax0_0_244 = 0; ax0_0_244 < 4; ++ax0_0_244) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_244 * 512)) + (ax3_0_1_122 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_244 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_244 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_244 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_244 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_245 = 0; ax0_0_245 < 4; ++ax0_0_245) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_245 * 512)) + (ax3_0_1_122 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_245 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_245 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_245 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_245 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_122 = 0; ax1_0_3_122 < 4; ++ax1_0_3_122) {
      for (int ax2_0_3_122 = 0; ax2_0_3_122 < 4; ++ax2_0_3_122) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_122 * 32) + (ax2_0_3_122 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_122 * 32) + (ax2_0_3_122 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_122 * 32) + (ax2_0_3_122 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_122 * 32) + (ax2_0_3_122 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_122 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_122 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_122 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_122 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_122 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_122 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_122 * 32) + (ax2_0_3_122 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_122 * 32) + (ax2_0_3_122 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_122 * 32) + (ax2_0_3_122 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_122 * 32) + (ax2_0_3_122 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_122 * 32) + (ax2_0_3_122 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_122 * 32) + (ax2_0_3_122 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_122 * 32) + (ax2_0_3_122 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_122 * 32) + (ax2_0_3_122 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_122 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_122 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_122 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_122 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_122 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_122 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_122 * 32) + (ax2_0_3_122 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_122 * 32) + (ax2_0_3_122 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_122 * 32) + (ax2_0_3_122 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_122 * 32) + (ax2_0_3_122 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_252 = 0; ax0_ax1_fused_0_252 < 4; ++ax0_ax1_fused_0_252) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_252 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_252 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 4032))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_253 = 0; ax0_ax1_fused_0_253 < 4; ++ax0_ax1_fused_0_253) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_253 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_253 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 4032))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_123 = 0; ax3_0_1_123 < 2; ++ax3_0_1_123) {
    for (int ax0_0_246 = 0; ax0_0_246 < 4; ++ax0_0_246) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_246 * 512)) + (ax3_0_1_123 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_246 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_246 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_246 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_246 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_247 = 0; ax0_0_247 < 4; ++ax0_0_247) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_247 * 512)) + (ax3_0_1_123 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_247 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_247 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_247 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_247 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_123 = 0; ax1_0_3_123 < 4; ++ax1_0_3_123) {
      for (int ax2_0_3_123 = 0; ax2_0_3_123 < 4; ++ax2_0_3_123) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_123 * 32) + (ax2_0_3_123 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_123 * 32) + (ax2_0_3_123 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_123 * 32) + (ax2_0_3_123 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_123 * 32) + (ax2_0_3_123 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_123 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_123 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_123 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_123 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_123 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_123 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_123 * 32) + (ax2_0_3_123 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_123 * 32) + (ax2_0_3_123 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_123 * 32) + (ax2_0_3_123 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_123 * 32) + (ax2_0_3_123 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_123 * 32) + (ax2_0_3_123 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_123 * 32) + (ax2_0_3_123 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_123 * 32) + (ax2_0_3_123 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_123 * 32) + (ax2_0_3_123 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_123 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_123 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_123 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_123 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_123 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_123 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_123 * 32) + (ax2_0_3_123 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_123 * 32) + (ax2_0_3_123 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_123 * 32) + (ax2_0_3_123 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_123 * 32) + (ax2_0_3_123 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_254 = 0; ax0_ax1_fused_0_254 < 4; ++ax0_ax1_fused_0_254) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_254 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 57344));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_254 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 4064))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_255 = 0; ax0_ax1_fused_0_255 < 4; ++ax0_ax1_fused_0_255) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_255 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 31) * 524288) + (ax0_ax1_fused_0_255 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 4064))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int ax3_0_1_124 = 0; ax3_0_1_124 < 2; ++ax3_0_1_124) {
    for (int ax0_0_248 = 0; ax0_0_248 < 4; ++ax0_0_248) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_248 * 512)) + (ax3_0_1_124 * 16)) + 16384)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_248 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_248 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_248 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0_248 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_249 = 0; ax0_0_249 < 4; ++ax0_0_249) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 2048) + (ax0_0_249 * 512)) + (ax3_0_1_124 * 16))])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_249 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_249 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_249 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax0_0_249 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_124 = 0; ax1_0_3_124 < 4; ++ax1_0_3_124) {
      for (int ax2_0_3_124 = 0; ax2_0_3_124 < 4; ++ax2_0_3_124) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_124 * 32) + (ax2_0_3_124 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_124 * 32) + (ax2_0_3_124 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_124 * 32) + (ax2_0_3_124 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_124 * 32) + (ax2_0_3_124 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_124 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_124 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_124 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_124 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_124 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + (ax2_0_3_124 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_124 * 32) + (ax2_0_3_124 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_124 * 32) + (ax2_0_3_124 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_124 * 32) + (ax2_0_3_124 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_124 * 32) + (ax2_0_3_124 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_124 * 32) + (ax2_0_3_124 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_124 * 32) + (ax2_0_3_124 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_124 * 32) + (ax2_0_3_124 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_124 * 32) + (ax2_0_3_124 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_124 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_124 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_124 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3_124 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_124 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp + ((ax2_0_3_124 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_124 * 32) + (ax2_0_3_124 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_124 * 32) + (ax2_0_3_124 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_124 * 32) + (ax2_0_3_124 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_124 * 32) + (ax2_0_3_124 * 8)) + 4)))[3]));
  }
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  for (int ax3_0_1_125 = 0; ax3_0_1_125 < 2; ++ax3_0_1_125) {
    for (int ax0_0_250 = 0; ax0_0_250 < 4; ++ax0_0_250) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_250 * 512)) + (ax3_0_1_125 * 16)) + 20480)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_250 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_250 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_250 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_250 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_251 = 0; ax0_0_251 < 4; ++ax0_0_251) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_251 * 512)) + (ax3_0_1_125 * 16)) + 4096)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax0_0_251 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax0_0_251 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax0_0_251 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax0_0_251 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_125 = 0; ax1_0_3_125 < 4; ++ax1_0_3_125) {
      for (int ax2_0_3_125 = 0; ax2_0_3_125 < 4; ++ax2_0_3_125) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_125 * 32) + (ax2_0_3_125 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_125 * 32) + (ax2_0_3_125 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_125 * 32) + (ax2_0_3_125 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_125 * 32) + (ax2_0_3_125 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_125 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_125 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_125 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_125 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax2_0_3_125 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax2_0_3_125 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_125 * 32) + (ax2_0_3_125 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_125 * 32) + (ax2_0_3_125 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_125 * 32) + (ax2_0_3_125 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_125 * 32) + (ax2_0_3_125 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_125 * 32) + (ax2_0_3_125 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_125 * 32) + (ax2_0_3_125 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_125 * 32) + (ax2_0_3_125 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_125 * 32) + (ax2_0_3_125 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_125 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_125 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_125 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_125 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + ((ax2_0_3_125 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + ((ax2_0_3_125 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_125 * 32) + (ax2_0_3_125 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_125 * 32) + (ax2_0_3_125 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_125 * 32) + (ax2_0_3_125 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_125 * 32) + (ax2_0_3_125 * 8)) + 4)))[3]));
  }
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  for (int ax3_0_1_126 = 0; ax3_0_1_126 < 2; ++ax3_0_1_126) {
    for (int ax0_0_252 = 0; ax0_0_252 < 4; ++ax0_0_252) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_252 * 512)) + (ax3_0_1_126 * 16)) + 24576)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_252 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_252 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_252 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_252 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_253 = 0; ax0_0_253 < 4; ++ax0_0_253) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_253 * 512)) + (ax3_0_1_126 * 16)) + 8192)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax0_0_253 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax0_0_253 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax0_0_253 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax0_0_253 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_126 = 0; ax1_0_3_126 < 4; ++ax1_0_3_126) {
      for (int ax2_0_3_126 = 0; ax2_0_3_126 < 4; ++ax2_0_3_126) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_126 * 32) + (ax2_0_3_126 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_126 * 32) + (ax2_0_3_126 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_126 * 32) + (ax2_0_3_126 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_126 * 32) + (ax2_0_3_126 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_126 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_126 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_126 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_126 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax2_0_3_126 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax2_0_3_126 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_126 * 32) + (ax2_0_3_126 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_126 * 32) + (ax2_0_3_126 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_126 * 32) + (ax2_0_3_126 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_126 * 32) + (ax2_0_3_126 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_126 * 32) + (ax2_0_3_126 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_126 * 32) + (ax2_0_3_126 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_126 * 32) + (ax2_0_3_126 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_126 * 32) + (ax2_0_3_126 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_126 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_126 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_126 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_126 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + ((ax2_0_3_126 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + ((ax2_0_3_126 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_126 * 32) + (ax2_0_3_126 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_126 * 32) + (ax2_0_3_126 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_126 * 32) + (ax2_0_3_126 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_126 * 32) + (ax2_0_3_126 * 8)) + 4)))[3]));
  }
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int ax3_0_1_127 = 0; ax3_0_1_127 < 2; ++ax3_0_1_127) {
    for (int ax0_0_254 = 0; ax0_0_254 < 4; ++ax0_0_254) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 2048) + (ax0_0_254 * 512)) + (ax3_0_1_127 * 16)) + 28672)])) + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_254 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_254 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_254 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_254 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_255 = 0; ax0_0_255 < 4; ++ax0_0_255) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 2048) + (ax0_0_255 * 512)) + (ax3_0_1_127 * 16)) + 12288)])) + ((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 8)));
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax0_0_255 * 8)))[0]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax0_0_255 * 8)))[1]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax0_0_255 * 8)))[2]), "=r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax0_0_255 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_127 = 0; ax1_0_3_127 < 4; ++ax1_0_3_127) {
      for (int ax2_0_3_127 = 0; ax2_0_3_127 < 4; ++ax2_0_3_127) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_127 * 32) + (ax2_0_3_127 * 8))))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_127 * 32) + (ax2_0_3_127 * 8))))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_127 * 32) + (ax2_0_3_127 * 8))))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_127 * 32) + (ax2_0_3_127 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_127 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_127 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_127 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_127 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax2_0_3_127 * 8)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + (ax2_0_3_127 * 8)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_127 * 32) + (ax2_0_3_127 * 8))))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_127 * 32) + (ax2_0_3_127 * 8))))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_127 * 32) + (ax2_0_3_127 * 8))))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + ((ax1_0_3_127 * 32) + (ax2_0_3_127 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_127 * 32) + (ax2_0_3_127 * 8)) + 4)))[0]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_127 * 32) + (ax2_0_3_127 * 8)) + 4)))[1]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_127 * 32) + (ax2_0_3_127 * 8)) + 4)))[2]), "=f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_127 * 32) + (ax2_0_3_127 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_127 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_127 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_127 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_127 * 8)))[3]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + ((ax2_0_3_127 * 8) + 4)))[0]), "r"(((unsigned *)(T_transpose_reindex_shared_dyn_warp_1 + ((ax2_0_3_127 * 8) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_127 * 32) + (ax2_0_3_127 * 8)) + 4)))[0]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_127 * 32) + (ax2_0_3_127 * 8)) + 4)))[1]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_127 * 32) + (ax2_0_3_127 * 8)) + 4)))[2]), "f"(((float *)(matmul_reindex_shared_dyn_warp + (((ax1_0_3_127 * 32) + (ax2_0_3_127 * 8)) + 4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax1_1 = 0; ax1_1 < 4; ++ax1_1) {
    for (int ax2_1 = 0; ax2_1 < 4; ++ax2_1) {
      for (int local_id = 0; local_id < 8; ++local_id) {
        ((float*)buf_dyn_shmem)[(((((((((((int)threadIdx.z) * 8192) + (ax1_1 * 2048)) + (((local_id & 3) >> 1) * 1024)) + ((((int)threadIdx.x) >> 2) * 128)) + (((int)threadIdx.y) * 64)) + (ax2_1 * 16)) + ((local_id >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (local_id & 1))] = matmul_reindex_shared_dyn_warp[(((ax1_1 * 32) + (ax2_1 * 8)) + local_id)];
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_256 = 0; ax0_ax1_fused_0_256 < 16; ++ax0_ax1_fused_0_256) {
    uint4 __1;
    ulonglong4 v_ = *(ulonglong4*)(((float*)buf_dyn_shmem) + ((((ax0_ax1_fused_0_256 * 1024) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)));
    ((half2*)(&(__1.x)))->x = (half)(((float2*)(&(v_.x)))->x);
    ((half2*)(&(__1.x)))->y = (half)(((float2*)(&(v_.x)))->y);
    ((half2*)(&(__1.y)))->x = (half)(((float2*)(&(v_.y)))->x);
    ((half2*)(&(__1.y)))->y = (half)(((float2*)(&(v_.y)))->y);
    ((half2*)(&(__1.z)))->x = (half)(((float2*)(&(v_.z)))->x);
    ((half2*)(&(__1.z)))->y = (half)(((float2*)(&(v_.z)))->y);
    ((half2*)(&(__1.w)))->x = (half)(((float2*)(&(v_.w)))->x);
    ((half2*)(&(__1.w)))->y = (half)(((float2*)(&(v_.w)))->y);
    *(uint4*)(compute + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_256 * 32768)) + (((int)threadIdx.z) * 16384)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 15) * 8))) = __1;
  }
}


