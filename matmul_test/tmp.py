
def shared_16x16_to_ldmatrix_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 4 * (j // 8) + (i // 8) * 2 + (j % 2)


for i in range(16):
    for j in range(16):
        print(shared_16x16_to_ldmatrix_32x8_layout(i, j), end="\t")
    print("")
