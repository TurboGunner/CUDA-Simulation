#include "matrix.cuh"

__host__ __device__ Matrix::Matrix(const size_t& rows_in, const size_t& columns_in, const bool& local) {
    rows = rows_in;
    columns = columns_in;

    is_square = rows == columns;

    cudaError_t cuda_status = cudaSuccess;

    size_t size_alloc = rows * columns * sizeof(float);

#ifdef __CUDA_ARCH__
    cuda_status = cudaMalloc(&data_device, size_alloc);
    if (cuda_status != cudaSuccess) {
        printf("%s", "Error: Did not properly allocate matrix (device, on device).");
    }
#else
    cuda_status = cudaMallocHost(&data, size_alloc);
    CudaExceptionHandler(cuda_status, "Could not allocate the memory for the matrix (host).");
    if (!local) {
        cuda_status = cudaMalloc(&data_device, size_alloc);
        CudaExceptionHandler(cuda_status, "Could not allocate the memory for the matrix (device, on host).");
    }
#endif
}


__host__ __device__ Matrix* Matrix::Create(const size_t& rows, const size_t& columns, const bool& local) {
    Matrix* matrix;
    cudaError_t cuda_status = cudaSuccess;

#ifdef __CUDA_ARCH__
    cuda_status = cudaMalloc(&matrix, sizeof(Matrix));
    if (cuda_status != cudaSuccess) {
        printf("%s", "Error: Did not properly allocate matrix pointer (device, on device).");
    }
#else
    cuda_status = cudaMallocHost(&matrix, sizeof(Matrix));
    CudaExceptionHandler(cuda_status, "Could not allocate the memory for the matrix pointer (host).");
    if (!local) {
        cuda_status = cudaMalloc(&matrix, sizeof(Matrix));
        CudaExceptionHandler(cuda_status, "Could not allocate the memory for the matrix pointer (device, on host).");
    }
#endif
    *matrix = Matrix(rows, columns, local);

    return matrix;
}

__host__ __device__ size_t Matrix::IX(size_t row, size_t column) const {
    //printf("IX: %zu\n", column + (row * rows));
    return column + (row * rows);
}

__host__ __device__ Matrix Matrix::Transpose() {
    Matrix output(columns, rows);
    for (size_t i = 0; i < columns; i++) {
        for (size_t j = 0; j < rows; j++) {
            output[output.IX(i, j)] = Get(IX(j, i));
    }
}
    return output;
    }

__host__ __device__ float& Matrix::Get(const int& index) {
    if (index >= IX(rows, columns) || index < 0) {
        printf("%s\n", "Warning: Out of bounds!");
#ifdef __CUDA_ARCH__
        return data_device[0];
#else
        return data[0];
#endif
    }
#ifdef __CUDA_ARCH__
    return data_device[index];
#else
    return data[index];
#endif
}

__host__ __device__ float& Matrix::Get(const size_t& row, const size_t& column) {
    return Get(IX(row, column));
}

__host__ __device__ float& Matrix::operator[](const int& index) {
    return Get(index);
}

__host__ __device__ void Matrix::Set(const float& value, const int& index) {
    if (index >= IX(rows, columns) || index < 0) {
        printf("%s\n", "Warning: Out of bounds!");
        return;
    }
#ifdef __CUDA_ARCH__
    data_device[index] = value;
#else
    data[index] = value;
#endif
}

__host__ __device__ void Matrix::Set(const float& value, const size_t& row, const size_t& column) {
    Set(value, IX(row, column));
}

__host__ __device__  void Matrix::GetCofactor(Matrix& output_matrix, int p, int q, int n) {
    int i = 0, j = 0;
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            if (row == p || col == q) {
                continue;
            }
            output_matrix[IX(i, j++)] = Get(IX(row, col));

            if (j != n - 1) {
                continue;
            }
            j = 0;
            i++;
        }
    }
}

__host__ __device__ int Matrix::Determinant(Matrix& matrix, size_t length) {
    int determinant = 0;

    if (length == 1) {
        return matrix.Get(0);
    }

    Matrix temp = Matrix(length, length);

    int sign = 1;

    for (int f = 0; f < length; f++) {
        matrix.GetCofactor(temp, 0, f, length);
        determinant += sign * matrix[matrix.IX(0, f)] * Determinant(temp, length - 1);

        sign = -sign;
    }

    return determinant;
}

__host__ __device__ void Matrix::Adjoint(Matrix& matrix, Matrix& adjoint) {
    if (matrix.rows == 1) {
        adjoint[0] = 1;
        return;
    }

    int sign = 1;
    Matrix temp = Matrix(matrix.rows, matrix.rows);

    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.rows; j++) {
            matrix.GetCofactor(temp, i, j, matrix.rows);

            sign = ((i + j) % 2 == 0) ? 1 : -1;

            adjoint[adjoint.IX(j, i)] = (sign) * (Determinant(temp, matrix.rows - 1));
        }
    }
}

__host__ __device__ bool Matrix::Inverse(Matrix& matrix, Matrix& inverse) {

    int determinant = Determinant(matrix, matrix.rows);

    if (determinant == 0) {
        printf("%s\n", "Singular matrix or non-square matrix, can't find its inverse");
        return false;
    }

    Matrix adjoint(matrix.rows, matrix.rows);
    Adjoint(matrix, adjoint);

    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.rows; j++) {
            inverse[inverse.IX(i, j)] = adjoint[adjoint.IX(i, j)] / (float) determinant;
        }
    }
    return true;
}