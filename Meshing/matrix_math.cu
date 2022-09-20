#include "matrix.cuh"

__host__ __device__ Matrix Matrix::Transpose() {
    Matrix output(columns, rows);
    for (size_t i = 0; i < columns; i++) {
        for (size_t j = 0; j < rows; j++) {
            output[output.IX(i, j)] = Get(IX(j, i));
        }
    }
    return output;
}

__host__ __device__  void Matrix::GetCofactor(Matrix& output_matrix, int p, int q, int n) {
    int i = 0, j = 0;
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            if (row == p || col == q) {
                continue;
            }
            output_matrix.Get(IX(i, j++)) = Get(IX(row, col));
            if (j != n - 1) {
                continue;
            }
            j = 0;
            i++;
        }
    }
}

__host__ __device__ float Matrix::Determinant(Matrix& matrix, size_t length) {
    float determinant = 0.0f;

    if (length == 1) {
        return matrix.Get(0);
    }

    Matrix temp(length, length);

    if (determinant > 0) {
        printf("AAAAA %zu\n", length);
    }

    float sign = 1.0f;

    for (int f = 0; f < length; f++) {
        matrix.GetCofactor(temp, 0, f, length);
        float pre_calculation = matrix.Get(matrix.IX(0, f)) * Determinant(temp, length - 1);
        determinant += sign * pre_calculation;

        sign = -sign;
    }

    return determinant;
}

__host__ __device__ void Matrix::Adjoint(Matrix& matrix, Matrix& adjoint) {
    if (matrix.rows == 1) {
        adjoint[0] = 1.0f;
        return;
    }

    float sign = 1.0f;
    Matrix temp(matrix.rows, matrix.rows);

    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.rows; j++) {
            matrix.GetCofactor(temp, i, j, matrix.rows);

            sign = ((i + j) % 2 == 0) ? 1 : -1;

            adjoint.Get(j, i) = (sign) * (Determinant(temp, matrix.rows - 1));
        }
    }
}

__host__ __device__ bool Matrix::Inverse(Matrix& matrix, Matrix& inverse) {
    float determinant = Determinant(matrix, matrix.rows);

    if (determinant == 0) {
        printf("%s\n", "Singular matrix, can't find its inverse");
        return false;
    }

    Matrix adjoint(matrix.rows, matrix.rows);
    Adjoint(matrix, adjoint);

    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.rows; j++) {
            inverse.Get(i, j) = adjoint.Get(i, j) / determinant;
        }
    }
    return true;
}

__host__ __device__ Matrix* Matrix::DiagonalMatrix(const float* points, const size_t& rows, const size_t& columns) {
    Matrix* output = Matrix::Create(rows, columns);
    for (int i = 0; i < rows; i++) {
         output->Get(i, i) = points[i];
    }

    return output;
}

__host__ __device__ Matrix Matrix::AbsoluteValue() {
    Matrix output(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            output.Get(i, j) = abs(Get(i, j));
        }
    }

    return output;
}

__host__ __device__ Matrix Matrix::Reciprocal() {
    Matrix output(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            float value_idx = Get(i, j);
            output.Get(i, j) = value_idx != 0 ? (1.0f / (value_idx)) : 0;
        }
    }

    return output;
}

__host__ __device__ Matrix Matrix::operator*(const float& scalar) {
    Matrix matrix(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            matrix.Get(i, j) = Get(i, j) * scalar;
        }
    }
    return matrix;
}

__host__ __device__ void Matrix::AddOnPointer(Matrix* matrix, Matrix add) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            matrix->Get(i, j) += add.Get(i, j);
        }
    }
}

__host__ __device__ void Matrix::MultiplyScalarOnPointer(Matrix* matrix, const float& multi) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            matrix->Get(i, j) *= multi;
        }
    }
}