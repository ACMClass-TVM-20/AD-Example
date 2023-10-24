#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

// 包装函数
#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long
void fused_decode3_wrapper(torch::Tensor lv2608, torch::Tensor lv2609, torch::Tensor lv7330,
                           torch::Tensor p_output0_intermediate, int64_t b, int64_t blocksize);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_decode3", &fused_decode3_wrapper, "Fused decode3 function");
}
