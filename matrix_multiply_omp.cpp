#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>

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

vector<vector<double>> multiply_matrices_omp(const vector<vector<double>> &a, const vector<vector<double>> &b) {
    int n = a.size();
    int m = a[0].size();
    int p = b[0].size();
    
    if (m != b.size()) {
        cerr << "Error: Matrices cannot be multiplied. Columns of A must equal rows of B." << endl;
        exit(1);
    }

    vector<vector<double>> result(n, vector<double>(p, 0.0));
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            double sum = 0.0;
            for (int k = 0; k < m; ++k) {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
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
    omp_set_num_threads(omp_get_max_threads());
    auto start = high_resolution_clock::now();

    auto matrix1 = read_matrix("matrix1.txt");
    auto matrix2 = read_matrix("matrix2.txt");

    auto multiply_start = high_resolution_clock::now();
    auto result = multiply_matrices_omp(matrix1, matrix2);
    auto multiply_end = high_resolution_clock::now();

    double multiply_time = duration_cast<duration<double>>(multiply_end - multiply_start).count();

    size_t bytes = 2 * 1000 * 1000 * sizeof(double);
    double data_size = static_cast<double>(bytes) / (1024 * 1024);

    save_result(result, "result_omp.txt", multiply_time, data_size);

    auto end = high_resolution_clock::now();
    double total_time = duration_cast<duration<double>>(end - start).count();
    cout << "Total execution time: " << total_time << " seconds" << endl;

    return 0;
}