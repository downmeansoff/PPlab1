import numpy as np
import time

def read_matrix(filename):
    return np.loadtxt(filename)

def main():
    try:
        with open('result_omp.txt', 'r') as f:
            lines = f.readlines()
        matrix_lines = [line.strip() for line in lines if not line.startswith('Time') and not line.startswith('Data')]
        result_cpp = np.loadtxt(matrix_lines)
        
        time_line = [line for line in lines if line.startswith('Time')][0]
        data_size_line = [line for line in lines if line.startswith('Data')][0]
        
        matrix1 = read_matrix('matrix1.txt')
        matrix2 = read_matrix('matrix2.txt')
        
        start = time.time()
        result_np = np.dot(matrix1, matrix2)
        np_time = time.time() - start
        
        print(f"NumPy multiplication time: {np_time:.4f} seconds")
        print(f"OpenMP time from file: {time_line.split(': ')[1].strip()}")
        
        if np.allclose(result_cpp, result_np, rtol=1e-5, atol=1e-8):
            print("Verification successful: Results are within tolerance.")
        else:
            print("Verification failed: Results differ.")
            
    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    main()