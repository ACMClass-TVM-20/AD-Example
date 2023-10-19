import os
import subprocess


shapes = [
    # linear layer various batch
    [1, 512, 4096, 4096],
    [4, 512, 4096, 4096],
    [6, 512, 4096, 4096],
    [10, 512, 4096, 4096],
    # MLP linear layer
    [4, 512, 4096, 11008],
    [4, 512, 11008, 4096],
    [1, 512, 4096, 11008],
    [1, 512, 11008, 4096],
    [10, 512, 4096, 11008],
    [10, 512, 11008, 4096],
    # small k
    [1, 512, 128, 4096],
    [1, 512, 64, 4096],
    [1, 512, 16, 4096],
    [1, 512, 8, 4096],
    # small m
    [1, 128, 4096, 4096],
    [1, 64, 4096, 4096],
    [1, 16, 4096, 4096],
    [1, 8, 4096, 4096],
    # # small n
    [1, 512, 4096, 128],
    [1, 512, 4096, 64],
    [1, 512, 4096, 16],
    [1, 512, 4096, 8],
    # all small
    [1, 128, 32, 128],
    [1, 16, 16, 16],
    [1, 8, 8, 8],
    # not divisible
    [6, 513, 4097, 4097],
    [2, 500, 4000, 4000],
    [1, 128, 32, 127],
    [1, 128, 32, 129],
]

cur_path = os.path.dirname(os.path.abspath(__file__))
before_path = os.path.join(cur_path, "test-3-tool.py")

for shape in shapes:
    command = f"python3 {before_path} {shape[0]} {shape[1]} {shape[2]} {shape[3]}"

    result = subprocess.run(command.split())

    if result.returncode != 0:
        pass
        # print(f"Command '{command}' failed with error code {result.returncode}:")
        # break
    print("")
