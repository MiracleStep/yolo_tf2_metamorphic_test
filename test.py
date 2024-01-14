import numpy as np

# 假设你有一个矩阵 A
A = np.array([[1, -1, 1],
              [1, -1, -1],
              [1, -1, 2]])

# 计算 A 的加号逆
B = np.linalg.pinv(A)

print("Matrix A:")
print(A)

print("Pseudoinverse of A:")
print(B*14)