#include "matrix.cuh"

__host__ __device__ Matrix::Matrix(const size_t& rows_in, const size_t& columns_in, const bool& local_in, const bool& host_in) {
    rows = rows_in;
    columns = columns_in;

    cudaError_t cuda_status = cudaSuccess;

    is_square = rows == columns;

    size_t size_alloc = rows * columns * sizeof(float);

    local = local_in;

#ifdef __CUDA_ARCH__
    data_device = (float*) malloc(size_alloc);
#else
    if (host_in) {
        cuda_status = cudaMallocHost(&data, size_alloc);
        CudaExceptionHandler(cuda_status, "Could not allocate the memory for the matrix (host).");
    }
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
    matrix = new Matrix(rows, columns, true, false);

    matrix->device_alloc = matrix;
#else
    cuda_status = cudaMallocHost(&matrix, sizeof(Matrix));
    CudaExceptionHandler(cuda_status, "Could not allocate the memory for the matrix pointer (host).");
    *matrix = Matrix(rows, columns, local);
    if (!local) {
        cuda_status = cudaMalloc(&(matrix->device_alloc), sizeof(Matrix));
        CudaExceptionHandler(cuda_status, "Could not allocate the memory for the matrix pointer (device, on host).");
        matrix->DeviceTransfer(matrix);
    }
#endif
    return matrix;
}

__host__ __device__ size_t Matrix::IX(size_t row, size_t column) const {
    return column + (rows * row);
}

__host__ __device__ float& Matrix::Get(const int& index) {
    if (index >= rows * columns || index < 0 || rows * columns == 0) {
        printf("%s%d\n", "Warning: Out of bounds (MatrixGet)! Index: ", index);
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
        printf("%s%d\n", "Warning: Out of bounds (MatrixSet)! Index: ", index);
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

__host__ cudaError_t Matrix::HostTransfer() {
    cudaError_t cuda_status = cudaSuccess;
    cuda_status = CopyFunction("HostTransferTable", data, data_device, cudaMemcpyDeviceToHost, cuda_status, sizeof(float), rows * columns);
    cudaDeviceSynchronize();

    return cuda_status;
}

__host__ cudaError_t Matrix::DeviceTransfer(Matrix* src) {
    cudaError_t cuda_status = cudaSuccess;

    if (!device_allocated_status) {
        cuda_status = cudaMalloc(&device_alloc, sizeof(Matrix));
        device_allocated_status = true;
        cuda_status = CopyFunction("DeviceTransferObject", device_alloc, src, cudaMemcpyHostToDevice, cuda_status, sizeof(Matrix), 1);
        if (local) {
            std::cout << "Warning! Host locality is set to true, but device synchronization was called." <<
                "\n\nThis will likely result in a segfault, as the corresponding GPU table data was not initialized." << std::endl;
        }
    }

    cuda_status = CopyFunction("DeviceTransferTable", data_device, data, cudaMemcpyHostToDevice, cuda_status, sizeof(float), rows * columns);
    cudaDeviceSynchronize();

    return cuda_status;
}

__host__ __device__ void Matrix::PrintMatrix(const char* label) {
    if (label) { // Nullptr default check
        printf("\n\n%s", label);
    }
    else {
        printf("\n\n");
    }
    for (size_t i = 0; i < rows; i++) {
        printf("\n");
        for (size_t j = 0; j < columns; j++) {
            printf("%f ", Get(IX(j, i)));
        }
    }
}

__host__ __device__ float* Matrix::Row(const size_t& index) {
    float* output = (float*) malloc(columns * sizeof(float));
    for (int i = 0; i < columns; i++) {
        output[i] = Get(i, index);
    }
    return output;
}

__host__ __device__ float* Matrix::Column(const size_t& index) {
    float* output = (float*)malloc(rows * sizeof(float));
    for (int i = 0; i < rows; i++) {
        output[i] = Get(index, i);
    }
    return output;
}

__host__ __device__ cudaError_t Matrix::Destroy() {
    cudaError_t cuda_status = cudaSuccess;
#ifndef __CUDA_ARCH__
    if (!local) {
        cuda_status = cudaFree(data_device);
        if (cuda_status != cudaSuccess) {
            printf("%s\n", "Could not free memory for the data device.");
        }
    }
    if (device_allocated_status) {
        cuda_status = cudaFree(device_alloc);
        printf("%s\n", "Could not free memory for the device allocation.");
    }
#endif
    free(data);
    //printf("%s\n", "Destruction of matrix successful.");
    return cuda_status;
}

__host__ void Matrix::DeleteAllocations(vector<Matrix*> matrices) {
    for (auto matrix : matrices) {
        matrix->Destroy();
    }
}

cudaError_t Matrix::PopulateRandomHost(Matrix* matrix, const float& min, const float& max) {
    RandomFloat random(min, max, 3);

    for (size_t i = 0; i < matrix->rows; i++) {
        for (size_t j = 0; j < matrix->columns; j++) {
            matrix->Set(random.Generate(), j, i);
        }
    }

    return matrix->DeviceTransfer(matrix);
}

__host__ __device__ Matrix* Matrix::MatrixMassAllocation(const size_t& size, const size_t& rows, const size_t& columns) {
    cudaError_t cuda_status = cudaSuccess;

    Matrix* output = nullptr;
    cuda_status = cudaMalloc(&output, sizeof(Matrix) * size * rows * columns);
    thrust::device_ptr<Matrix> thrust_alloc(output);

    thrust::fill(thrust_alloc, thrust_alloc + size, Matrix(rows, columns, false, false));

    return thrust_alloc.get();
}

__host__ __device__ void Matrix::CopyMatrixOnPointer(Matrix* matrix, Matrix& copy) {
    for (size_t i = 0; i < matrix->rows; i++) {
        for (size_t j = 0; j < matrix->columns; j++) {
            matrix->Get(j, i) = copy.Get(j, i);
        }
    }
}