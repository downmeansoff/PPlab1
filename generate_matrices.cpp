#include <iostream>
#include <fstream>
#include <random>
#include <vector>

using namespace std;

vector<vector<double>> generate_matrix(int rows, int cols) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 100.0);

    vector<vector<double>> matrix(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

void save_matrix(const vector<vector<double>> &matrix, const string &filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Could not open file " << filename << endl;
        exit(1);
    }
    for (const auto &row : matrix) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << row[j];
            if (j != row.size() - 1) file << " ";
        }
        file << endl;
    }
    file.close();
}

int main() {
    int n = 1000;
    auto matrix1 = generate_matrix(n, n);
    auto matrix2 = generate_matrix(n, n);
    save_matrix(matrix1, "matrix1.txt");
    save_matrix(matrix2, "matrix2.txt");
    return 0;
}