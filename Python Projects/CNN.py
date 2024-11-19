import numpy as np

# Simple Convolution, ReLU, and Pooling in a compact form

# Example 5x5 Image and 3x3 Kernel
image = np.array(
    [
        [1, 2, 0, 3, 1],
        [4, 0, 1, 1, 0],
        [1, 3, 2, 2, 4],
        [0, 1, 1, 3, 1],
        [2, 4, 1, 0, 0],
    ]
)

kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

# Convolution
conv = np.array(
    [[np.sum(image[i : i + 3, j : j + 3] * kernel) for j in range(3)] for i in range(3)]
)
print("Convolution:\n", conv)

# ReLU
relu = np.maximum(conv, 0)
print("\nReLU:\n", relu)

# Max Pooling
pool = np.array(
    [
        [np.max(relu[i : i + 2, j : j + 2]) for j in range(0, 2, 2)]
        for i in range(0, 2, 2)
    ]
)
print("\nMax Pooling:\n", pool)
