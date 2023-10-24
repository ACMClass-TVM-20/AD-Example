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
extern "C" __global__ void __launch_bounds__(128) fused_fused_decode3_fused_NT_matmul21_cast21_add5_silu2_kernel(uint* __restrict__ lv2608, half* __restrict__ lv2609, half* __restrict__ p_output0_intermediate_1) {
  extern __shared__ uchar buf_dyn_shmem[];
  uint2 __1;
    uint2 __2;
      uint2 __3;
      uint4 __4;
        uint4 __5;
          uint4 v_ = make_uint4(lv2608[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))], lv2608[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))], lv2608[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))], lv2608[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]);
          uint4 v__1 = make_uint4(((uint)0)+((uint)4*0), ((uint)0)+((uint)4*1), ((uint)0)+((uint)4*2), ((uint)0)+((uint)4*3));
          __5.x = (v_.x >> v__1.x);
          __5.y = (v_.y >> v__1.y);
          __5.z = (v_.z >> v__1.z);
          __5.w = (v_.w >> v__1.w);
        uint4 v__2 = make_uint4((uint)15, (uint)15, (uint)15, (uint)15);
        __4.x = (__5.x & v__2.x);
        __4.y = (__5.y & v__2.y);
        __4.z = (__5.z & v__2.z);
        __4.w = (__5.w & v__2.w);
      ((half2*)(&(__3.x)))->x = (half)(__4.x);
      ((half2*)(&(__3.x)))->y = (half)(__4.y);
      ((half2*)(&(__3.y)))->x = (half)(__4.z);
      ((half2*)(&(__3.y)))->y = (half)(__4.w);
      uint2 v__3 = make_uint2(__pack_half2(__float2half_rn(7.000000e+00f), __float2half_rn(7.000000e+00f)), __pack_half2(__float2half_rn(7.000000e+00f), __float2half_rn(7.000000e+00f)));
      ((half2*)(&(__2.x)))->x = (((half2*)(&(__3.x)))->x-((half2*)(&(v__3.x)))->x);
      ((half2*)(&(__2.x)))->y = (((half2*)(&(__3.x)))->y-((half2*)(&(v__3.x)))->y);
      ((half2*)(&(__2.y)))->x = (((half2*)(&(__3.y)))->x-((half2*)(&(v__3.y)))->x);
      ((half2*)(&(__2.y)))->y = (((half2*)(&(__3.y)))->y-((half2*)(&(v__3.y)))->y);
    uint2 v__4 = make_uint2(__pack_half2(lv2609[((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 2))], lv2609[((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 2))]), __pack_half2(lv2609[((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 2))], lv2609[((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 2))]));
    ((half2*)(&(__1.x)))->x = (((half2*)(&(__2.x)))->x*((half2*)(&(v__4.x)))->x);
    ((half2*)(&(__1.x)))->y = (((half2*)(&(__2.x)))->y*((half2*)(&(v__4.x)))->y);
    ((half2*)(&(__1.y)))->x = (((half2*)(&(__2.y)))->x*((half2*)(&(v__4.y)))->x);
    ((half2*)(&(__1.y)))->y = (((half2*)(&(__2.y)))->y*((half2*)(&(v__4.y)))->y);
  *(uint2*)(p_output0_intermediate_1 + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 8))) = __1;
  uint2 __6;
    uint2 __7;
      uint2 __8;
      uint4 __9;
        uint4 __10;
          uint4 v__5 = make_uint4(lv2608[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))], lv2608[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))], lv2608[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))], lv2608[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]);
          uint4 v__6 = make_uint4(((uint)16)+((uint)4*0), ((uint)16)+((uint)4*1), ((uint)16)+((uint)4*2), ((uint)16)+((uint)4*3));
          __10.x = (v__5.x >> v__6.x);
          __10.y = (v__5.y >> v__6.y);
          __10.z = (v__5.z >> v__6.z);
          __10.w = (v__5.w >> v__6.w);
        uint4 v__7 = make_uint4((uint)15, (uint)15, (uint)15, (uint)15);
        __9.x = (__10.x & v__7.x);
        __9.y = (__10.y & v__7.y);
        __9.z = (__10.z & v__7.z);
        __9.w = (__10.w & v__7.w);
      ((half2*)(&(__8.x)))->x = (half)(__9.x);
      ((half2*)(&(__8.x)))->y = (half)(__9.y);
      ((half2*)(&(__8.y)))->x = (half)(__9.z);
      ((half2*)(&(__8.y)))->y = (half)(__9.w);
      uint2 v__8 = make_uint2(__pack_half2(__float2half_rn(7.000000e+00f), __float2half_rn(7.000000e+00f)), __pack_half2(__float2half_rn(7.000000e+00f), __float2half_rn(7.000000e+00f)));
      ((half2*)(&(__7.x)))->x = (((half2*)(&(__8.x)))->x-((half2*)(&(v__8.x)))->x);
      ((half2*)(&(__7.x)))->y = (((half2*)(&(__8.x)))->y-((half2*)(&(v__8.x)))->y);
      ((half2*)(&(__7.y)))->x = (((half2*)(&(__8.y)))->x-((half2*)(&(v__8.y)))->x);
      ((half2*)(&(__7.y)))->y = (((half2*)(&(__8.y)))->y-((half2*)(&(v__8.y)))->y);
    uint2 v__9 = make_uint2(__pack_half2(lv2609[((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 2))], lv2609[((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 2))]), __pack_half2(lv2609[((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 2))], lv2609[((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 2))]));
    ((half2*)(&(__6.x)))->x = (((half2*)(&(__7.x)))->x*((half2*)(&(v__9.x)))->x);
    ((half2*)(&(__6.x)))->y = (((half2*)(&(__7.x)))->y*((half2*)(&(v__9.x)))->y);
    ((half2*)(&(__6.y)))->x = (((half2*)(&(__7.y)))->x*((half2*)(&(v__9.y)))->x);
    ((half2*)(&(__6.y)))->y = (((half2*)(&(__7.y)))->y*((half2*)(&(v__9.y)))->y);
  *(uint2*)(p_output0_intermediate_1 + (((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 8)) + 4)) = __6;
}

extern "C" __global__ void __launch_bounds__(128) fused_fused_decode3_fused_NT_matmul21_cast21_add5_silu2_kernel_1(half* __restrict__ lv7330, half* __restrict__ p_output0_intermediate, half* __restrict__ p_output0_intermediate_1, int64_t b) {
  extern __shared__ uchar buf_dyn_shmem[];
  float var_NT_matmul_intermediate_reindex_shared_dyn_warp[128];
  half lv7330_reindex_shared_dyn_warp[32];
  half p_output0_intermediate_1_reindex_shared_dyn_warp[32];
  half lv7330_reindex_shared_dyn_warp_1[32];
  half p_output0_intermediate_1_reindex_shared_dyn_warp_1[32];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 4; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 4; ++ax2_0_3_init) {
      for (int i = 0; i < 8; ++i) {
var_NT_matmul_intermediate_reindex_shared_dyn_warp[((ax1_0_3_init * 32) + (ax2_0_3_init * 8)) + i] = 0.0;}
;
    }
  }
  for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)4; ++ax0_ax1_fused_0) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((((ax0_ax1_fused_0 * (int64_t)2048) + (((int64_t)threadIdx.z) * (int64_t)1024)) + (((int64_t)threadIdx.y) * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)64)) + (((((int64_t)threadIdx.x) & (int64_t)3) ^ (((int64_t)threadIdx.x) >> (int64_t)3)) * (int64_t)16)) + (int64_t)32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(lv7330 + (((((((((int64_t)blockIdx.x) / (int64_t)86) * (int64_t)524288) + (ax0_ax1_fused_0 * (int64_t)131072)) + (((int64_t)threadIdx.z) * (int64_t)65536)) + (((int64_t)threadIdx.y) * (int64_t)32768)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)4096)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)))), "n"(16)
    );
  }
  }
  for (int64_t ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < (int64_t)4; ++ax0_ax1_fused_0_1) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_1 * (int64_t)2048) + (((int64_t)threadIdx.z) * (int64_t)1024)) + (((int64_t)threadIdx.y) * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)64)) + (((((int64_t)threadIdx.x) & (int64_t)3) ^ (((int64_t)threadIdx.x) >> (int64_t)3)) * (int64_t)16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(p_output0_intermediate_1 + (((((((((int64_t)blockIdx.x) % (int64_t)86) * (int64_t)524288) + (ax0_ax1_fused_0_1 * (int64_t)131072)) + (((int64_t)threadIdx.z) * (int64_t)65536)) + (((int64_t)threadIdx.y) * (int64_t)32768)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)4096)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int64_t ax0_ax1_fused_0_2 = 0; ax0_ax1_fused_0_2 < (int64_t)4; ++ax0_ax1_fused_0_2) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((((ax0_ax1_fused_0_2 * (int64_t)2048) + (((int64_t)threadIdx.z) * (int64_t)1024)) + (((int64_t)threadIdx.y) * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)64)) + (((((int64_t)threadIdx.x) & (int64_t)3) ^ (((int64_t)threadIdx.x) >> (int64_t)3)) * (int64_t)16)) + (int64_t)40960));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(lv7330 + ((((((((((int64_t)blockIdx.x) / (int64_t)86) * (int64_t)524288) + (ax0_ax1_fused_0_2 * (int64_t)131072)) + (((int64_t)threadIdx.z) * (int64_t)65536)) + (((int64_t)threadIdx.y) * (int64_t)32768)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)4096)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)) + (int64_t)32))), "n"(16)
    );
  }
  }
  for (int64_t ax0_ax1_fused_0_3 = 0; ax0_ax1_fused_0_3 < (int64_t)4; ++ax0_ax1_fused_0_3) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((((ax0_ax1_fused_0_3 * (int64_t)2048) + (((int64_t)threadIdx.z) * (int64_t)1024)) + (((int64_t)threadIdx.y) * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)64)) + (((((int64_t)threadIdx.x) & (int64_t)3) ^ (((int64_t)threadIdx.x) >> (int64_t)3)) * (int64_t)16)) + (int64_t)8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(p_output0_intermediate_1 + ((((((((((int64_t)blockIdx.x) % (int64_t)86) * (int64_t)524288) + (ax0_ax1_fused_0_3 * (int64_t)131072)) + (((int64_t)threadIdx.z) * (int64_t)65536)) + (((int64_t)threadIdx.y) * (int64_t)32768)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)4096)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)) + (int64_t)32))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int64_t ax0_ax1_fused_0_4 = 0; ax0_ax1_fused_0_4 < (int64_t)4; ++ax0_ax1_fused_0_4) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((((ax0_ax1_fused_0_4 * (int64_t)2048) + (((int64_t)threadIdx.z) * (int64_t)1024)) + (((int64_t)threadIdx.y) * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)64)) + (((((int64_t)threadIdx.x) & (int64_t)3) ^ (((int64_t)threadIdx.x) >> (int64_t)3)) * (int64_t)16)) + (int64_t)49152));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(lv7330 + ((((((((((int64_t)blockIdx.x) / (int64_t)86) * (int64_t)524288) + (ax0_ax1_fused_0_4 * (int64_t)131072)) + (((int64_t)threadIdx.z) * (int64_t)65536)) + (((int64_t)threadIdx.y) * (int64_t)32768)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)4096)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)) + (int64_t)64))), "n"(16)
    );
  }
  }
  for (int64_t ax0_ax1_fused_0_5 = 0; ax0_ax1_fused_0_5 < (int64_t)4; ++ax0_ax1_fused_0_5) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((((ax0_ax1_fused_0_5 * (int64_t)2048) + (((int64_t)threadIdx.z) * (int64_t)1024)) + (((int64_t)threadIdx.y) * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)64)) + (((((int64_t)threadIdx.x) & (int64_t)3) ^ (((int64_t)threadIdx.x) >> (int64_t)3)) * (int64_t)16)) + (int64_t)16384));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(p_output0_intermediate_1 + ((((((((((int64_t)blockIdx.x) % (int64_t)86) * (int64_t)524288) + (ax0_ax1_fused_0_5 * (int64_t)131072)) + (((int64_t)threadIdx.z) * (int64_t)65536)) + (((int64_t)threadIdx.y) * (int64_t)32768)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)4096)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)) + (int64_t)64))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)125; ++ax3_0_0) {
    __syncthreads();
    for (int64_t ax0_ax1_fused_0_6 = 0; ax0_ax1_fused_0_6 < (int64_t)4; ++ax0_ax1_fused_0_6) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((((((ax3_0_0 + (int64_t)3) & (int64_t)3) * (int64_t)8192) + (ax0_ax1_fused_0_6 * (int64_t)2048)) + (((int64_t)threadIdx.z) * (int64_t)1024)) + (((int64_t)threadIdx.y) * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)64)) + (((((int64_t)threadIdx.x) & (int64_t)3) ^ (((int64_t)threadIdx.x) >> (int64_t)3)) * (int64_t)16)) + (int64_t)32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(lv7330 + (((((((((((int64_t)blockIdx.x) / (int64_t)86) * (int64_t)524288) + (ax0_ax1_fused_0_6 * (int64_t)131072)) + (((int64_t)threadIdx.z) * (int64_t)65536)) + (((int64_t)threadIdx.y) * (int64_t)32768)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)4096)) + (ax3_0_0 * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)) + (int64_t)96))), "n"(16)
    );
  }
    }
    for (int64_t ax0_ax1_fused_0_7 = 0; ax0_ax1_fused_0_7 < (int64_t)4; ++ax0_ax1_fused_0_7) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((((((ax3_0_0 + (int64_t)3) & (int64_t)3) * (int64_t)8192) + (ax0_ax1_fused_0_7 * (int64_t)2048)) + (((int64_t)threadIdx.z) * (int64_t)1024)) + (((int64_t)threadIdx.y) * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)64)) + (((((int64_t)threadIdx.x) & (int64_t)3) ^ (((int64_t)threadIdx.x) >> (int64_t)3)) * (int64_t)16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(p_output0_intermediate_1 + (((((((((((int64_t)blockIdx.x) % (int64_t)86) * (int64_t)524288) + (ax0_ax1_fused_0_7 * (int64_t)131072)) + (((int64_t)threadIdx.z) * (int64_t)65536)) + (((int64_t)threadIdx.y) * (int64_t)32768)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)4096)) + (ax3_0_0 * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)) + (int64_t)96))), "n"(16)
    );
  }
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

    __syncthreads();
    for (int64_t ax3_0_1 = 0; ax3_0_1 < (int64_t)2; ++ax3_0_1) {
      for (int64_t ax0_0 = 0; ax0_0 < (int64_t)4; ++ax0_0) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((((ax3_0_0 & (int64_t)3) * (int64_t)4096) + (((int64_t)threadIdx.z) * (int64_t)2048)) + (ax0_0 * (int64_t)512)) + ((((int64_t)threadIdx.x) & (int64_t)15) * (int64_t)32)) + ((((ax3_0_1 * (int64_t)2) + (((int64_t)threadIdx.x) >> (int64_t)4)) ^ ((((int64_t)threadIdx.x) & (int64_t)7) >> (int64_t)1)) * (int64_t)8)) + (int64_t)16384)])) + (int64_t)0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(lv7330_reindex_shared_dyn_warp + (ax0_0 * (int64_t)8)))[0]), "=r"(((unsigned *)(lv7330_reindex_shared_dyn_warp + (ax0_0 * (int64_t)8)))[1]), "=r"(((unsigned *)(lv7330_reindex_shared_dyn_warp + (ax0_0 * (int64_t)8)))[2]), "=r"(((unsigned *)(lv7330_reindex_shared_dyn_warp + (ax0_0 * (int64_t)8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int64_t ax0_0_1 = 0; ax0_0_1 < (int64_t)4; ++ax0_0_1) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((((ax3_0_0 & (int64_t)3) * (int64_t)4096) + (((int64_t)threadIdx.y) * (int64_t)2048)) + (ax0_0_1 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)4) * (int64_t)256)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)32)) + ((((ax3_0_1 * (int64_t)2) + ((((int64_t)threadIdx.x) & (int64_t)15) >> (int64_t)3)) ^ ((((int64_t)threadIdx.x) & (int64_t)7) >> (int64_t)1)) * (int64_t)8))])) + (int64_t)0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp + (ax0_0_1 * (int64_t)8)))[0]), "=r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp + (ax0_0_1 * (int64_t)8)))[1]), "=r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp + (ax0_0_1 * (int64_t)8)))[2]), "=r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp + (ax0_0_1 * (int64_t)8)))[3])
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
      :  "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)ax2_0_3) * (int64_t)8))))[0]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)ax2_0_3) * (int64_t)8))))[1]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)ax2_0_3) * (int64_t)8))))[2]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)ax2_0_3) * (int64_t)8))))[3])
      : "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp + (((int64_t)ax1_0_3) * (int64_t)8)))[0]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp + (((int64_t)ax1_0_3) * (int64_t)8)))[1]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp + (((int64_t)ax1_0_3) * (int64_t)8)))[2]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp + (((int64_t)ax1_0_3) * (int64_t)8)))[3]), "r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp + (((int64_t)ax2_0_3) * (int64_t)8)))[0]), "r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp + (((int64_t)ax2_0_3) * (int64_t)8)))[1]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)ax2_0_3) * (int64_t)8))))[0]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)ax2_0_3) * (int64_t)8))))[1]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)ax2_0_3) * (int64_t)8))))[2]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)ax2_0_3) * (int64_t)8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)ax2_0_3) * (int64_t)8)) + (int64_t)4)))[0]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)ax2_0_3) * (int64_t)8)) + (int64_t)4)))[1]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)ax2_0_3) * (int64_t)8)) + (int64_t)4)))[2]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)ax2_0_3) * (int64_t)8)) + (int64_t)4)))[3])
      : "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp + (((int64_t)ax1_0_3) * (int64_t)8)))[0]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp + (((int64_t)ax1_0_3) * (int64_t)8)))[1]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp + (((int64_t)ax1_0_3) * (int64_t)8)))[2]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp + (((int64_t)ax1_0_3) * (int64_t)8)))[3]), "r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp + ((((int64_t)ax2_0_3) * (int64_t)8) + (int64_t)4)))[0]), "r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp + ((((int64_t)ax2_0_3) * (int64_t)8) + (int64_t)4)))[1]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)ax2_0_3) * (int64_t)8)) + (int64_t)4)))[0]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)ax2_0_3) * (int64_t)8)) + (int64_t)4)))[1]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)ax2_0_3) * (int64_t)8)) + (int64_t)4)))[2]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)ax2_0_3) * (int64_t)8)) + (int64_t)4)))[3]));
  }
        }
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  for (int64_t ax3_0_1_1 = 0; ax3_0_1_1 < (int64_t)2; ++ax3_0_1_1) {
    for (int64_t ax0_0_2 = 0; ax0_0_2 < (int64_t)4; ++ax0_0_2) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((((int64_t)threadIdx.z) * (int64_t)2048) + (ax0_0_2 * (int64_t)512)) + ((((int64_t)threadIdx.x) & (int64_t)15) * (int64_t)32)) + ((((ax3_0_1_1 * (int64_t)2) + (((int64_t)threadIdx.x) >> (int64_t)4)) ^ ((((int64_t)threadIdx.x) & (int64_t)7) >> (int64_t)1)) * (int64_t)8)) + (int64_t)20480)])) + (int64_t)0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (ax0_0_2 * (int64_t)8)))[0]), "=r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (ax0_0_2 * (int64_t)8)))[1]), "=r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (ax0_0_2 * (int64_t)8)))[2]), "=r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (ax0_0_2 * (int64_t)8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int64_t ax0_0_3 = 0; ax0_0_3 < (int64_t)4; ++ax0_0_3) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.y) * (int64_t)2048) + (ax0_0_3 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)4) * (int64_t)256)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)32)) + ((((ax3_0_1_1 * (int64_t)2) + ((((int64_t)threadIdx.x) & (int64_t)15) >> (int64_t)3)) ^ ((((int64_t)threadIdx.x) & (int64_t)7) >> (int64_t)1)) * (int64_t)8)) + (int64_t)4096)])) + (int64_t)0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (ax0_0_3 * (int64_t)8)))[0]), "=r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (ax0_0_3 * (int64_t)8)))[1]), "=r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (ax0_0_3 * (int64_t)8)))[2]), "=r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (ax0_0_3 * (int64_t)8)))[3])
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
      :  "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax2_0_3_1) * (int64_t)8))))[0]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax2_0_3_1) * (int64_t)8))))[1]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax2_0_3_1) * (int64_t)8))))[2]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax2_0_3_1) * (int64_t)8))))[3])
      : "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_1) * (int64_t)8)))[0]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_1) * (int64_t)8)))[1]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_1) * (int64_t)8)))[2]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_1) * (int64_t)8)))[3]), "r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (((int64_t)ax2_0_3_1) * (int64_t)8)))[0]), "r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (((int64_t)ax2_0_3_1) * (int64_t)8)))[1]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax2_0_3_1) * (int64_t)8))))[0]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax2_0_3_1) * (int64_t)8))))[1]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax2_0_3_1) * (int64_t)8))))[2]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax2_0_3_1) * (int64_t)8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax2_0_3_1) * (int64_t)8)) + (int64_t)4)))[0]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax2_0_3_1) * (int64_t)8)) + (int64_t)4)))[1]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax2_0_3_1) * (int64_t)8)) + (int64_t)4)))[2]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax2_0_3_1) * (int64_t)8)) + (int64_t)4)))[3])
      : "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_1) * (int64_t)8)))[0]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_1) * (int64_t)8)))[1]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_1) * (int64_t)8)))[2]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_1) * (int64_t)8)))[3]), "r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + ((((int64_t)ax2_0_3_1) * (int64_t)8) + (int64_t)4)))[0]), "r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + ((((int64_t)ax2_0_3_1) * (int64_t)8) + (int64_t)4)))[1]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax2_0_3_1) * (int64_t)8)) + (int64_t)4)))[0]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax2_0_3_1) * (int64_t)8)) + (int64_t)4)))[1]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax2_0_3_1) * (int64_t)8)) + (int64_t)4)))[2]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax2_0_3_1) * (int64_t)8)) + (int64_t)4)))[3]));
  }
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  for (int64_t ax3_0_1_2 = 0; ax3_0_1_2 < (int64_t)2; ++ax3_0_1_2) {
    for (int64_t ax0_0_4 = 0; ax0_0_4 < (int64_t)4; ++ax0_0_4) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((((int64_t)threadIdx.z) * (int64_t)2048) + (ax0_0_4 * (int64_t)512)) + ((((int64_t)threadIdx.x) & (int64_t)15) * (int64_t)32)) + ((((ax3_0_1_2 * (int64_t)2) + (((int64_t)threadIdx.x) >> (int64_t)4)) ^ ((((int64_t)threadIdx.x) & (int64_t)7) >> (int64_t)1)) * (int64_t)8)) + (int64_t)24576)])) + (int64_t)0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (ax0_0_4 * (int64_t)8)))[0]), "=r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (ax0_0_4 * (int64_t)8)))[1]), "=r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (ax0_0_4 * (int64_t)8)))[2]), "=r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (ax0_0_4 * (int64_t)8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int64_t ax0_0_5 = 0; ax0_0_5 < (int64_t)4; ++ax0_0_5) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.y) * (int64_t)2048) + (ax0_0_5 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)4) * (int64_t)256)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)32)) + ((((ax3_0_1_2 * (int64_t)2) + ((((int64_t)threadIdx.x) & (int64_t)15) >> (int64_t)3)) ^ ((((int64_t)threadIdx.x) & (int64_t)7) >> (int64_t)1)) * (int64_t)8)) + (int64_t)8192)])) + (int64_t)0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (ax0_0_5 * (int64_t)8)))[0]), "=r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (ax0_0_5 * (int64_t)8)))[1]), "=r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (ax0_0_5 * (int64_t)8)))[2]), "=r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (ax0_0_5 * (int64_t)8)))[3])
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
      :  "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_2) * (int64_t)32) + (((int64_t)ax2_0_3_2) * (int64_t)8))))[0]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_2) * (int64_t)32) + (((int64_t)ax2_0_3_2) * (int64_t)8))))[1]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_2) * (int64_t)32) + (((int64_t)ax2_0_3_2) * (int64_t)8))))[2]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_2) * (int64_t)32) + (((int64_t)ax2_0_3_2) * (int64_t)8))))[3])
      : "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_2) * (int64_t)8)))[0]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_2) * (int64_t)8)))[1]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_2) * (int64_t)8)))[2]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_2) * (int64_t)8)))[3]), "r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (((int64_t)ax2_0_3_2) * (int64_t)8)))[0]), "r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (((int64_t)ax2_0_3_2) * (int64_t)8)))[1]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_2) * (int64_t)32) + (((int64_t)ax2_0_3_2) * (int64_t)8))))[0]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_2) * (int64_t)32) + (((int64_t)ax2_0_3_2) * (int64_t)8))))[1]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_2) * (int64_t)32) + (((int64_t)ax2_0_3_2) * (int64_t)8))))[2]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_2) * (int64_t)32) + (((int64_t)ax2_0_3_2) * (int64_t)8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_2) * (int64_t)32) + (((int64_t)ax2_0_3_2) * (int64_t)8)) + (int64_t)4)))[0]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_2) * (int64_t)32) + (((int64_t)ax2_0_3_2) * (int64_t)8)) + (int64_t)4)))[1]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_2) * (int64_t)32) + (((int64_t)ax2_0_3_2) * (int64_t)8)) + (int64_t)4)))[2]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_2) * (int64_t)32) + (((int64_t)ax2_0_3_2) * (int64_t)8)) + (int64_t)4)))[3])
      : "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_2) * (int64_t)8)))[0]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_2) * (int64_t)8)))[1]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_2) * (int64_t)8)))[2]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_2) * (int64_t)8)))[3]), "r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + ((((int64_t)ax2_0_3_2) * (int64_t)8) + (int64_t)4)))[0]), "r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + ((((int64_t)ax2_0_3_2) * (int64_t)8) + (int64_t)4)))[1]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_2) * (int64_t)32) + (((int64_t)ax2_0_3_2) * (int64_t)8)) + (int64_t)4)))[0]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_2) * (int64_t)32) + (((int64_t)ax2_0_3_2) * (int64_t)8)) + (int64_t)4)))[1]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_2) * (int64_t)32) + (((int64_t)ax2_0_3_2) * (int64_t)8)) + (int64_t)4)))[2]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_2) * (int64_t)32) + (((int64_t)ax2_0_3_2) * (int64_t)8)) + (int64_t)4)))[3]));
  }
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int64_t ax3_0_1_3 = 0; ax3_0_1_3 < (int64_t)2; ++ax3_0_1_3) {
    for (int64_t ax0_0_6 = 0; ax0_0_6 < (int64_t)4; ++ax0_0_6) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((((int64_t)threadIdx.z) * (int64_t)2048) + (ax0_0_6 * (int64_t)512)) + ((((int64_t)threadIdx.x) & (int64_t)15) * (int64_t)32)) + ((((ax3_0_1_3 * (int64_t)2) + (((int64_t)threadIdx.x) >> (int64_t)4)) ^ ((((int64_t)threadIdx.x) & (int64_t)7) >> (int64_t)1)) * (int64_t)8)) + (int64_t)28672)])) + (int64_t)0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (ax0_0_6 * (int64_t)8)))[0]), "=r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (ax0_0_6 * (int64_t)8)))[1]), "=r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (ax0_0_6 * (int64_t)8)))[2]), "=r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (ax0_0_6 * (int64_t)8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int64_t ax0_0_7 = 0; ax0_0_7 < (int64_t)4; ++ax0_0_7) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.y) * (int64_t)2048) + (ax0_0_7 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)4) * (int64_t)256)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)32)) + ((((ax3_0_1_3 * (int64_t)2) + ((((int64_t)threadIdx.x) & (int64_t)15) >> (int64_t)3)) ^ ((((int64_t)threadIdx.x) & (int64_t)7) >> (int64_t)1)) * (int64_t)8)) + (int64_t)12288)])) + (int64_t)0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (ax0_0_7 * (int64_t)8)))[0]), "=r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (ax0_0_7 * (int64_t)8)))[1]), "=r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (ax0_0_7 * (int64_t)8)))[2]), "=r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (ax0_0_7 * (int64_t)8)))[3])
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
      :  "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_3) * (int64_t)32) + (((int64_t)ax2_0_3_3) * (int64_t)8))))[0]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_3) * (int64_t)32) + (((int64_t)ax2_0_3_3) * (int64_t)8))))[1]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_3) * (int64_t)32) + (((int64_t)ax2_0_3_3) * (int64_t)8))))[2]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_3) * (int64_t)32) + (((int64_t)ax2_0_3_3) * (int64_t)8))))[3])
      : "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_3) * (int64_t)8)))[0]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_3) * (int64_t)8)))[1]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_3) * (int64_t)8)))[2]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_3) * (int64_t)8)))[3]), "r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (((int64_t)ax2_0_3_3) * (int64_t)8)))[0]), "r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + (((int64_t)ax2_0_3_3) * (int64_t)8)))[1]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_3) * (int64_t)32) + (((int64_t)ax2_0_3_3) * (int64_t)8))))[0]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_3) * (int64_t)32) + (((int64_t)ax2_0_3_3) * (int64_t)8))))[1]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_3) * (int64_t)32) + (((int64_t)ax2_0_3_3) * (int64_t)8))))[2]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + ((((int64_t)ax1_0_3_3) * (int64_t)32) + (((int64_t)ax2_0_3_3) * (int64_t)8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_3) * (int64_t)32) + (((int64_t)ax2_0_3_3) * (int64_t)8)) + (int64_t)4)))[0]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_3) * (int64_t)32) + (((int64_t)ax2_0_3_3) * (int64_t)8)) + (int64_t)4)))[1]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_3) * (int64_t)32) + (((int64_t)ax2_0_3_3) * (int64_t)8)) + (int64_t)4)))[2]), "=f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_3) * (int64_t)32) + (((int64_t)ax2_0_3_3) * (int64_t)8)) + (int64_t)4)))[3])
      : "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_3) * (int64_t)8)))[0]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_3) * (int64_t)8)))[1]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_3) * (int64_t)8)))[2]), "r"(((unsigned *)(lv7330_reindex_shared_dyn_warp_1 + (((int64_t)ax1_0_3_3) * (int64_t)8)))[3]), "r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + ((((int64_t)ax2_0_3_3) * (int64_t)8) + (int64_t)4)))[0]), "r"(((unsigned *)(p_output0_intermediate_1_reindex_shared_dyn_warp_1 + ((((int64_t)ax2_0_3_3) * (int64_t)8) + (int64_t)4)))[1]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_3) * (int64_t)32) + (((int64_t)ax2_0_3_3) * (int64_t)8)) + (int64_t)4)))[0]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_3) * (int64_t)32) + (((int64_t)ax2_0_3_3) * (int64_t)8)) + (int64_t)4)))[1]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_3) * (int64_t)32) + (((int64_t)ax2_0_3_3) * (int64_t)8)) + (int64_t)4)))[2]), "f"(((float *)(var_NT_matmul_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_3) * (int64_t)32) + (((int64_t)ax2_0_3_3) * (int64_t)8)) + (int64_t)4)))[3]));
  }
      }
    }
  }
  __syncthreads();
  for (int64_t ax1_1 = 0; ax1_1 < (int64_t)4; ++ax1_1) {
    for (int64_t ax2_1 = 0; ax2_1 < (int64_t)4; ++ax2_1) {
      for (int64_t local_id = 0; local_id < (int64_t)8; ++local_id) {
        ((float*)buf_dyn_shmem)[(((((((((int64_t)threadIdx.z) * (int64_t)8192) + (ax1_1 * (int64_t)2048)) + (((local_id & (int64_t)3) >> (int64_t)1) * (int64_t)1024)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)128)) + (((((((int64_t)threadIdx.y) * (int64_t)8) + (ax2_1 * (int64_t)2)) + (local_id >> (int64_t)2)) ^ (((int64_t)threadIdx.x) >> (int64_t)2)) * (int64_t)8)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)2)) + (local_id & (int64_t)1))] = var_NT_matmul_intermediate_reindex_shared_dyn_warp[(((ax1_1 * (int64_t)32) + (ax2_1 * (int64_t)8)) + local_id)];
      }
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0_8 = 0; ax0_ax1_fused_0_8 < (int64_t)16; ++ax0_ax1_fused_0_8) {
    uint4 __1;
    ulonglong4 v_ = *(ulonglong4*)(((float*)buf_dyn_shmem) + (((((ax0_ax1_fused_0_8 * (int64_t)1024) + (((int64_t)threadIdx.z) * (int64_t)512)) + (((int64_t)threadIdx.y) * (int64_t)256)) + ((((int64_t)threadIdx.x) >> (int64_t)4) * (int64_t)128)) + (((((int64_t)threadIdx.x) & (int64_t)15) ^ (((((int64_t)threadIdx.z) * (int64_t)4) + (((int64_t)threadIdx.y) * (int64_t)2)) + (((int64_t)threadIdx.x) >> (int64_t)4))) * (int64_t)8)));
    ((half2*)(&(__1.x)))->x = (half)(((float2*)(&(v_.x)))->x);
    ((half2*)(&(__1.x)))->y = (half)(((float2*)(&(v_.x)))->y);
    ((half2*)(&(__1.y)))->x = (half)(((float2*)(&(v_.y)))->x);
    ((half2*)(&(__1.y)))->y = (half)(((float2*)(&(v_.y)))->y);
    ((half2*)(&(__1.z)))->x = (half)(((float2*)(&(v_.z)))->x);
    ((half2*)(&(__1.z)))->y = (half)(((float2*)(&(v_.z)))->y);
    ((half2*)(&(__1.w)))->x = (half)(((float2*)(&(v_.w)))->x);
    ((half2*)(&(__1.w)))->y = (half)(((float2*)(&(v_.w)))->y);
    *(uint4*)(p_output0_intermediate + ((((((((((int64_t)blockIdx.x) / (int64_t)86) * (int64_t)1409024) + (ax0_ax1_fused_0_8 * (int64_t)88064)) + (((int64_t)threadIdx.z) * (int64_t)44032)) + (((int64_t)threadIdx.y) * (int64_t)22016)) + ((((int64_t)threadIdx.x) >> (int64_t)4) * (int64_t)11008)) + ((((int64_t)blockIdx.x) % (int64_t)86) * (int64_t)128)) + ((((int64_t)threadIdx.x) & (int64_t)15) * (int64_t)8))) = __1;
  }
}


