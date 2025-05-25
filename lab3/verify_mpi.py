import numpy as np

def read_matrix(filename, start):
    with open(filename, 'r') as f:
        lines = f.readlines()
    matrix = []
    for line in lines[start:]:
        matrix.append(list(map(float, line.strip().split())))
    return np.array(matrix)

A = read_matrix('A_750x400.txt', 1)
B = read_matrix('B_400x750.txt', 1)
C_correct = np.dot(A, B)

C_cpp = read_matrix('result.txt', 5)

if np.allclose(C_cpp, C_correct, atol=1e-5):
    print("Results match!")
else:
    print("Discrepancies found!")