for (int64_t ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 32; ++ax0_ax1_fused_0_1) {
  __syncthreads();
  ((half*)
       buf_dyn_shmem)[ax0_ax1_fused_0_1 * 128 + threadIdx.z * 64 + threadIdx.y * 32 +
                          ((threadIdx.x >> 3) ^ ((ax0_ax1_fused_0_1 & 1) * 2 + threadIdx.z)) * 8 +
                          (threadIdx.x) &
                      7] =
      ((half)((((uint*)buf_dyn_shmem)[ax0_ax1_fused_0_1 * 16 + threadIdx.z * 8 + threadIdx.y * 4 +
                                          threadIdx.x >>
                                      3 + 4096] >>
               (((uint)((threadIdx.x) & 7)) * (uint)4)) &
              (uint)15) -
       __float2half_rn(7.000000e+00f)) *
      ((half*)buf_dyn_shmem)[ax0_ax1_fused_0_1 * 4 + threadIdx.z * 2 + threadIdx.y + 9216];
}
