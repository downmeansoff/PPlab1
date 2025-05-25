#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <sstream>
#include <mpi.h>
#include <cstring>

using namespace std;
using namespace chrono;

typedef std::vector<vector<double>> Matrix;
const string PATH_MATRIX1 = "matrix/A_750x600.txt";
const string PATH_MATRIX2 = "matrix/B_600x750.txt";
const string PATH_RESULT = "result_mpi/750x600.txt";

void read_matrix(const string& filename, Matrix& matrix, int& rows, int& cols) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    string line;
    getline(file, line);
    istringstream size(line);
    size >> rows >> cols;

    matrix.resize(rows);
    
    int i = 0;
    while (getline(file, line)) {
        istringstream iss(line);
        double num;

        for(int j = 0; j < cols; j++) {
            iss >> num;
            matrix[i].push_back(num);
        }
        i++;

        if (i == rows) break;
    }
    file.close();
}

void write_matrix(const string& filename, const Matrix& matrix, const int& rows, 
                const int& cols, const double& time, const long long& volume) {
    ofstream out_file(filename);

    if (!out_file.is_open()) {
        cerr << "Error creating file!" << endl;
        return;
    }

    out_file << "rows: " << rows << endl;
    out_file << "columns: " << cols << endl;
    out_file << "time: " << time/1000 << " s" << endl;
    out_file << "count multiply: " << volume << "\n\n";

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out_file << matrix[i][j] << " ";
        }
        out_file << "\n";
    }

    out_file.close();
}

void distribute_matrix(const Matrix& matrix, double* buffer, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            buffer[i * cols + j] = matrix[i][j];
        }
    }
}

void gather_result(Matrix& result, const double* buffer, int rows, int cols, int my_rows) {
    result.resize(rows);
    for (int i = 0; i < rows; ++i) {
        result[i].resize(cols);
    }

    for (int i = 0; i < my_rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = buffer[i * cols + j];
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Matrix matrix1, matrix2;
    int rows1, cols1, rows2, cols2;

    // Процесс 0 читает матрицы и рассылает размеры
    if (rank == 0) {
        read_matrix(PATH_MATRIX1, matrix1, rows1, cols1);
        read_matrix(PATH_MATRIX2, matrix2, rows2, cols2);

        if (cols1 != rows2) {
            cerr << "Matrix dimensions mismatch!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Рассылаем размеры матриц всем процессам
    MPI_Bcast(&rows1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rows2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols2, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Вычисляем, сколько строк будет обрабатывать каждый процесс
    int rows_per_process = rows1 / size;
    int remainder = rows1 % size;
    int my_rows = (rank < remainder) ? rows_per_process + 1 : rows_per_process;
    int my_offset = (rank < remainder) ? rank * my_rows : 
                     remainder * (rows_per_process + 1) + 
                     (rank - remainder) * rows_per_process;

    // Рассылаем матрицу B всем процессам
    double* matrixB = new double[rows2 * cols2];
    if (rank == 0) {
        distribute_matrix(matrix2, matrixB, rows2, cols2);
    }
    MPI_Bcast(matrixB, rows2 * cols2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Распределяем части матрицы A между процессами
    double* my_matrixA = new double[my_rows * cols1];
    if (rank == 0) {
        // Процесс 0 заполняет свою часть и рассылает остальным
        double* full_matrixA = new double[rows1 * cols1];
        distribute_matrix(matrix1, full_matrixA, rows1, cols1);

        // Отправляем части другим процессам
        for (int p = 1; p < size; p++) {
            int p_rows = (p < remainder) ? rows_per_process + 1 : rows_per_process;
            int p_offset = (p < remainder) ? p * p_rows : 
                          remainder * (rows_per_process + 1) + 
                          (p - remainder) * rows_per_process;
            
            MPI_Send(full_matrixA + p_offset * cols1, p_rows * cols1, 
                    MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }

        // Копируем свою часть
        memcpy(my_matrixA, full_matrixA + my_offset * cols1, my_rows * cols1 * sizeof(double));
        delete[] full_matrixA;
    } else {
        // Получаем свою часть от процесса 0
        MPI_Recv(my_matrixA, my_rows * cols1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Вычисляем свою часть результата
    double* my_result = new double[my_rows * cols2];
    long long my_volume = 0;

    auto start = high_resolution_clock::now();
    
    for (int i = 0; i < my_rows; ++i) {
        for (int j = 0; j < cols2; ++j) {
            double sum = 0.0;
            for (int k = 0; k < cols1; ++k) {
                sum += my_matrixA[i * cols1 + k] * matrixB[k * cols2 + j];
                my_volume++;
            }
            my_result[i * cols2 + j] = sum;
        }
    }

    auto end = high_resolution_clock::now();
    auto time = duration_cast<milliseconds>(end - start).count();

    // Собираем результаты на процессе 0
    long long total_volume;
    MPI_Reduce(&my_volume, &total_volume, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double* full_result = nullptr;
    if (rank == 0) {
        full_result = new double[rows1 * cols2];
    }

    // Собираем части результата
    int* recv_counts = nullptr;
    int* displs = nullptr;

    if (rank == 0) {
        recv_counts = new int[size];
        displs = new int[size];

        for (int p = 0; p < size; p++) {
            recv_counts[p] = ((p < remainder) ? rows_per_process + 1 : rows_per_process) * cols2;
            displs[p] = ((p < remainder) ? p * (rows_per_process + 1) : 
                         remainder * (rows_per_process + 1) + 
                         (p - remainder) * rows_per_process) * cols2;
        }
    }

    MPI_Gatherv(my_result, my_rows * cols2, MPI_DOUBLE,
                full_result, recv_counts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Процесс 0 сохраняет результат
    if (rank == 0) {
        Matrix result;
        gather_result(result, full_result, rows1, cols2, rows1);
        
        cout << "rows: " << result.size() << endl;
        cout << "columns: " << result[0].size() << endl;
        cout << "time: " << time/1000.0 << " s" << endl;
        cout << "count multiply: " << total_volume << "\n\n";

        write_matrix(PATH_RESULT, result, result.size(), result[0].size(), time, total_volume);

        delete[] full_result;
        delete[] recv_counts;
        delete[] displs;
    }

    // Освобождаем память
    delete[] my_matrixA;
    delete[] matrixB;
    delete[] my_result;

    MPI_Finalize();
    return 0;
}
