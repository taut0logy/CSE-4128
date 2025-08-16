import numpy as np

# Create a sample 5x5 matrix
M = np.array([
    [2, 4, 1, 3, 5],
    [6, 5, 4, 2, 1],
    [7, 8, 9, 5, 4],
    [3, 1, 2, 4, 6],
    [8, 6, 4, 2, 0]
], dtype=float)

# Perform SVD
U, S, Vt = np.linalg.svd(M, full_matrices=False)

# Extract the first singular value and vectors
vector_x = U[:, [0]] * S[0]   # scale u1 by sigma1
vector_y = Vt[[0], :]         # 1x5 row vector

# Rank-1 approximation
M_approx = vector_x @ vector_y

print("Original Matrix M:\n", M)
print("\nvector_x (5x1):\n", vector_x)
print("\nvector_y (1x5):\n", vector_y)
print("\nReconstructed Matrix (Rank-1 Approximation):\n", M_approx)
