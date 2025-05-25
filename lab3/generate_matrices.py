import numpy as np

rows1, cols1 = 750, 500
matrix1 = np.random.randint(0, 10, size=(rows1, cols1))

rows2, cols2 = 500, 750
matrix2 = np.random.randint(0, 10, size=(rows2, cols2))

if cols1 != rows2:
    print("!!! COLS 1 != ROWS2 !!!")
    exit

with open("A_750x500.txt", "w") as file:
    file.write(f"{rows1} {cols1}\n")  # Записываем размер матрицы
    np.savetxt(file, matrix1, fmt="%d", delimiter=" ")  # Записываем саму матрицу

with open("B_500x750.txt", "w") as file:
    file.write(f"{rows2} {cols2}\n")  # Записываем размер матрицы
    np.savetxt(file, matrix2, fmt="%d", delimiter=" ")  # Записываем саму матрицу