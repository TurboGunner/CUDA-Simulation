#include "matrix.cuh"

__host__ __device__ Matrix::Matrix(const size_t& rows_in, const size_t& columns_in, const bool& local) {
    rows = rows_in;
    columns = columns_in;

    cudaError_t cuda_status = cudaSuccess;

    is_square = rows == columns;

    size_t size_alloc = rows * columns * sizeof(float);

#ifdef __CUDA_ARCH__
    cuda_status = cudaMalloc(&data_device, size_alloc);
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
    Matrix* matrix = nullptr;
    cudaError_t cuda_status = cudaSuccess;

#ifdef __CUDA_ARCH__
    cuda_status = cudaMalloc(&matrix->device_alloc, sizeof(Matrix));
    if (cuda_status != cudaSuccess) {
        printf("%s", "Error: Did not properly allocate matrix pointer (device, on device).");
    }
    *matrix = Matrix(rows, columns, local);
    *(matrix->device_alloc) = Matrix(rows, columns, local);
#else
    cuda_status = cudaMallocHost(&matrix, sizeof(Matrix));
    CudaExceptionHandler(cuda_status, "Could not allocate the memory for the matrix pointer (host).");
    *matrix = Matrix(rows, columns, local);
    if (!local) {
        cuda_status = cudaMalloc(&(matrix->device_alloc), sizeof(Matrix));
        CudaExceptionHandler(cuda_status, "Could not allocate the memory for the matrix pointer (device, on host).");
    }
#endif
    return matrix;
}

__host__ __device__ size_t Matrix::IX(size_t row, size_t column) const {
    return row + (rows * column);
}

__host__ __device__ float& Matrix::Get(const int& index) {
    if (index >= rows * columns || index < 0) {
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
    if (index >= rows * columns || index < 0) {
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
        //temp.PrintMatrix();
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

            //printf("%f\n", matrix[matrix.IX(j, i)]);

            adjoint[adjoint.IX(j, i)] = (sign) * (Determinant(temp, matrix.rows - 1));
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
            inverse.Get(inverse.IX(i, j)) = adjoint[adjoint.IX(i, j)] / determinant;
        }
    }
    return true;
}

__host__ cudaError_t Matrix::HostTransfer() {
    cudaError_t cuda_status = cudaSuccess;
    cuda_status = CopyFunction("HostTransferTable", data, data_device, cudaMemcpyDeviceToHost, cuda_status, sizeof(float), rows * columns);
    cudaDeviceSynchronize();

    return cuda_status;
}

__host__ cudaError_t Matrix::DeviceTransfer(Matrix* ptr, Matrix* src) {
    cudaError_t cuda_status = cudaSuccess;

    if (!device_allocated_status) {
        cuda_status = cudaMalloc(&ptr, sizeof(Matrix));
        device_allocated_status = true;
        cuda_status = CopyFunction("DeviceTransferObject", ptr, src, cudaMemcpyHostToDevice, cuda_status, sizeof(Matrix), 1);
        device_alloc = ptr;
    }
    else {
        ptr = device_alloc;
    }

    cuda_status = CopyFunction("DeviceTransferTable", data_device, data, cudaMemcpyHostToDevice, cuda_status, sizeof(float), rows * columns);
    cudaDeviceSynchronize();

    return cuda_status;
}

__host__ __device__ void Matrix::PrintMatrix() {
    for (size_t i = 0; i < rows; i++) {
        printf("\n");
        for (size_t j = 0; j < columns; j++) {
            printf("%f ", Get(IX(i, j)));
        }
    }
}