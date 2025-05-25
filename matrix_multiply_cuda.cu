#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

__global__ void matrixMultiply(double *A, double *B, double *C, int N) {
    __shared__ double tileA[32][32];
    __shared__ double tileB[32][32];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    for (int t = 0; t < (N + blockDim.x - 1)/blockDim.x; ++t) {
        if (row < N && (t*blockDim.x + threadIdx.x) < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row*N + t*blockDim.x + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < N && (t*blockDim.y + threadIdx.y) < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(t*blockDim.y + threadIdx.y)*N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int k = 0; k < blockDim.x; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row*N + col] = sum;
    }
}

vector<vector<double>> read_matrix(const string &filename) {
    ifstream file(filename);
    vector<vector<double>> matrix;
    string line;
    while (getline(file, line)) {
        vector<double> row;
        size_t pos = 0;
        while (pos < line.size()) {
            size_t end_pos = line.find(' ', pos);
            if (end_pos == string::npos) end_pos = line.size();
            string num_str = line.substr(pos, end_pos - pos);
            double num = stod(num_str);
            row.push_back(num);
            pos = end_pos + 1;
        }
        matrix.push_back(row);
    }
    return matrix;
}

vector<vector<double>> multiply_matrices_cuda(const vector<vector<double>> &a, const vector<vector<double>> &b) {
    int N = a.size();
    double *h_A = new double[N*N];
    double *h_B = new double[N*N];
    double *h_C = new double[N*N];

    // Конвертация матриц в плоские массивы
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i*N + j] = a[i][j];
            h_B[i*N + j] = b[i][j];
        }
    }

    
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*N*sizeof(double));
    cudaMalloc(&d_B, N*N*sizeof(double));
    cudaMalloc(&d_C, N*N*sizeof(double));

    cudaMemcpy(d_A, h_A, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);


    cudaMemcpy(h_C, d_C, N*N*sizeof(double), cudaMemcpyDeviceToHost);

    vector<vector<double>> result(N, vector<double>(N));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            result[i][j] = h_C[i*N + j];
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return result;
}

void save_result(const vector<vector<double>> &result, const string &filename, double duration, double data_size) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Could not open file " << filename << endl;
        exit(1);
    }
    for (const auto &row : result) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << row[j];
            if (j != row.size() - 1) file << " ";
        }
        file << endl;
    }
    file << "Time taken: " << duration << " seconds" << endl;
    file << "Data size: " << data_size << " MB" << endl;
    file.close();
}

int main() {
    auto start = high_resolution_clock::now();

    auto matrix1 = read_matrix("matrix1.txt");
    auto matrix2 = read_matrix("matrix2.txt");

    auto multiply_start = high_resolution_clock::now();
    auto result = multiply_matrices_cuda(matrix1, matrix2);
    auto multiply_end = high_resolution_clock::now();

    double multiply_time = duration_cast<duration<double>>(multiply_end - multiply_start).count();

    size_t bytes = 2 * 1000 * 1000 * sizeof(double);
    double data_size = static_cast<double>(bytes) / (1024 * 1024);

    save_result(result, "result_cuda.txt", multiply_time, data_size);

    auto end = high_resolution_clock::now();
    double total_time = duration_cast<duration<double>>(end - start).count();
    cout << "Total execution time: " << total_time << " seconds" << endl;

    return 0;
}