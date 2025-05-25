import numpy as np

def verify():
    try:
        res_mpi = np.loadtxt('result_mpi.txt', max_rows=1000)
        res_ref = np.loadtxt('result.txt', max_rows=1000)
        
        if np.allclose(res_mpi, res_ref, rtol=1e-5, atol=1e-8):
            print("MPI verification: Success")
        else:
            print("MPI verification: Failed")
            
    except Exception as e:
        print(f"Verification error: {e}")

if __name__ == "__main__":
    verify()