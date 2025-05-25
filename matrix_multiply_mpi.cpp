#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000;
    vector<vector<double>> matrixA, matrixB, result;
    double start_time, end_time;

    if (rank == 0) {
        matrixA = read_matrix("matrix1.txt");
        matrixB = read_matrix("matrix2.txt");
        result.resize(N, vector<double>(N));
        start_time = MPI_Wtime();
    }

    // Широковещательная рассылка матрицы B
    vector<double> B_flat(N*N);
    if (rank == 0) {
        for (int i = 0; i < N; ++i)
            copy(matrixB[i].begin(), matrixB[i].end(), B_flat.begin() + i*N);
    }
    MPI_Bcast(B_flat.data(), N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Распределение строк матрицы A
    int rows_per_proc = N / size;
    int extra_rows = N % size;
    vector<int> counts(size, rows_per_proc);
    vector<int> displs(size, 0);

    for (int i = 0; i < extra_rows; ++i) counts[i]++;
    for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + counts[i-1];

    vector<double> A_local(counts[rank] * N);
    MPI_Scatterv(rank == 0 ? matrixA[0].data() : nullptr,
                counts.data(), displs.data(), N*counts[0], MPI_DOUBLE,
                A_local.data(), counts[rank]*N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Локальное умножение
    vector<double> C_local(counts[rank] * N, 0.0);
    for (int i = 0; i < counts[rank]; ++i) {
        for (int k = 0; k < N; ++k) {
            double a = A_local[i*N + k];
            for (int j = 0; j < N; ++j) {
                C_local[i*N + j] += a * B_flat[k*N + j];
            }
        }
    }

    // Сбор результатов
    MPI_Gatherv(C_local.data(), counts[rank]*N, MPI_DOUBLE,
               rank == 0 ? result[0].data() : nullptr,
               counts.data(), displs.data(), MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        end_time = MPI_Wtime();
        ofstream file("result_mpi.txt");
        for (auto& row : result) {
            for (auto val : row) file << val << " ";
            file << endl;
        }
        file << "Time taken: " << end_time - start_time << " seconds" << endl;
        file << "Data size: " << (2*N*N*sizeof(double))/(1024*1024) << " MB" << endl;
    }

    MPI_Finalize();
    return 0;
}