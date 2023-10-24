import os
import subprocess


# size_tuples = [
#     (32000, 4096, 11008),
#     (32000, 5120, 13824),
#     (32000, 8192, 28672),
# ]

# m_values = [(1, 512), (4, 512), (10, 512), (-4, 512)]

# shapes = []

# for vocab_size, hidden_size, intermediate_size in size_tuples:
#     for batch_size, m_size in m_values:
#         shapes.append([batch_size, m_size, hidden_size, hidden_size])
#         shapes.append([batch_size, m_size, intermediate_size, hidden_size])
#         shapes.append([batch_size, m_size, hidden_size, intermediate_size])

# print("shapes:\n" + ",\n".join(map(str, shapes)) + "\n")

shapes = [
    # [1, 512, 4096, 4096],
    # [1, 512, 11008, 4096],
    # [1, 512, 4096, 11008],
    # [4, 512, 4096, 4096],
    # [4, 512, 11008, 4096],
    # [4, 512, 4096, 11008],
    # [10, 512, 4096, 4096],
    # [10, 512, 11008, 4096],
    # [10, 512, 4096, 11008],
    # [-4, 512, 4096, 4096],
    # [-4, 512, 11008, 4096],
    # [-4, 512, 4096, 11008],
    # [1, 512, 5120, 5120],
    # [1, 512, 13824, 5120],
    # [1, 512, 5120, 13824],
    # [4, 512, 5120, 5120],
    # [4, 512, 13824, 5120],
    # [4, 512, 5120, 13824],
    # [10, 512, 5120, 5120],
    # [10, 512, 13824, 5120],
    # [10, 512, 5120, 13824],
    # [-4, 512, 5120, 5120],
    # [-4, 512, 13824, 5120],
    # [-4, 512, 5120, 13824],
    # [1, 512, 8192, 8192],
    # [1, 512, 28672, 8192],
    # [1, 512, 8192, 28672],
    # [4, 512, 8192, 8192],
    # [4, 512, 28672, 8192],
    # [4, 512, 8192, 28672],
    # [10, 512, 8192, 8192],
    # [10, 512, 28672, 8192],
    # [10, 512, 8192, 28672],
    # [-4, 512, 8192, 8192],
    # [-4, 512, 28672, 8192],
    # [-4, 512, 8192, 28672],
    # [1, -512, 4096, 4096],
    [1, -512, 11008, 4096],
    [1, -512, 4096, 11008],
    [1, -1024, 4096, 4096],
    # [1, -1024, 11008, 4096],
    [1, -1024, 4096, 11008],
    [1, -2048, 4096, 4096],
    [1, -2048, 11008, 4096],
    [1, -2048, 4096, 11008],
]

cur_path = os.path.dirname(os.path.abspath(__file__))
before_path = os.path.join(cur_path, "test-6.py")

for shape in shapes:
    print(f"Testing shape: {shape}...")
    command = f"python3 {before_path} {shape[0]} {shape[1]} {shape[2]} {shape[3]}"

    result = subprocess.run(command.split())

    if result.returncode != 0:
        print(f"Command '{command}' failed with error code {result.returncode}:")
        pass
    print("")
