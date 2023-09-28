from tvm.contrib.nvcc import compile_cuda

file_name = "/home/yxdong/llm/tvm-develop/other-repos/AD-Example/matmul_test/build.cu"
compile_cuda(open(file_name, "r").read(), "cubin")
