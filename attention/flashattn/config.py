import torch
import tvm

# dtype
# dtype_torch = torch.float32
# dtype_str = "float32"
dtype_torch = torch.float16
dtype_str = "float16"

fallback_dtype_torch = torch.float32
fallback_dtype_str = "float32"

rtol, atol = (1e-6, 1e-3) if dtype_str == "float16" else (1e-6, 1e-3)


# device
# dev_torch = torch.device("cpu")
dev_torch = torch.device("mps")
# dev_torch = torch.device("cuda")

# target_tvm, dev_tvm = "llvm", tvm.cpu()
# target_tvm, dev_tvm = tvm.target.Target("cuda"), tvm.cuda()
target_tvm, dev_tvm = tvm.target.Target("apple/m1-gpu-restricted"), tvm.metal()

path_prefix = "flashattn"

# shapes
batch = 6
n_head = 32
seq = 512
head_dim = 128
Bc = 8
Br = 8

Tr = seq // Br
Tc = seq // Bc

# output config
enable_time_output = False
