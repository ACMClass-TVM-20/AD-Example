"""
Copyright © 2023 Apple Inc.

See LICENSE folder for this sample’s licensing information.

Abstract:
The code to run the compiled soft shrink kernel.
"""

# Allow soft shrink op to run through CPU fallback if it's not implemented.
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import time

import torch
from torch import nn

import torch.utils.cpp_extension

compiled_lib = torch.utils.cpp_extension.load(
    name="CustomSoftshrink",
    sources=["test.mm"],
    extra_cflags=["-std=c++17"],
)

assert torch.backends.mps.is_available()
mps_device = torch.device("mps")  # Device object representing GPU.


# Wrapper over the custom MPS soft shrink kernel.
class MPSSoftshrink(nn.Module):
    __constants__ = ["lambd"]
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, input):
        return compiled_lib.mps_softshrink(input, self.lambd)

    def extra_repr(self):
        return str(self.lambd)


# Wrapper over the Sequential layer, using the custom MPS kernel soft shrink implementation.
class CustomMPSSoftshrinkModel(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            MPSSoftshrink(),
            nn.Linear(lin1_size, lin2_size),
            MPSSoftshrink(),
            nn.Linear(lin2_size, lin3_size),
            MPSSoftshrink(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        return self.model(x)


# Wrapper over the Sequential layer, using the default soft shrink implementation.
class SoftshrinkModel(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.Softshrink(),
            nn.Linear(lin1_size, lin2_size),
            nn.Softshrink(),
            nn.Linear(lin2_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        return self.model(x)


# Tests the speedup of the custom soft shrink kernel.
def test_speedup():
    custom_mps_softshrink = 0
    default_softshrink = 0
    x = torch.randn(256, 784, device=mps_device)
    default_model = SoftshrinkModel().to(mps_device)
    custom_model = CustomMPSSoftshrinkModel().to(mps_device)

    # Measures time.
    for _ in range(100):
        start = time.time()
        default_model.forward(x)
        torch.mps.synchronize()
        default_softshrink += time.time() - start

        start = time.time()
        custom_model.forward(x)
        torch.mps.synchronize()
        custom_mps_softshrink += time.time() - start

    speedup = default_softshrink / custom_mps_softshrink
    print(
        "Default Softshrink: {:.3f} us | Custom Kernel MPS Softshrink {:.3f} us ({:.3f} times faster)".format(
            default_softshrink * 1e6 / 1e5, custom_mps_softshrink * 1e6 / 1e5, speedup
        )
    )


# Tests the correctness of the custom soft shrink kernel.
def test_correctness():
    custom_softshrink = MPSSoftshrink()
    default_softshrink = nn.Softshrink()

    input_data = torch.randn(256, 784, 326, device=mps_device, dtype=torch.float)

    output_custom_softshrink_op = custom_softshrink(input_data)
    output_default_softshrink_op = default_softshrink(input_data)

    torch.testing.assert_close(output_custom_softshrink_op, output_default_softshrink_op)


def test_softshrink():
    test_correctness()
    test_speedup()


if __name__ == "__main__":
    test_softshrink()
