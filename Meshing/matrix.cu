#include "matrix.cuh"

__host__ __device__ Matrix::Matrix(const size_t& rows_in, const size_t& columns_in, const bool& local_in) {
    rows = rows_in;
    columns = columns_in;

    cudaError_t cuda_status = cudaSuccess;

    is_square = rows == columns;

    size_t size_alloc = rows * columns * sizeof(float);

    local = local_in;

#ifdef __CUDA_ARCH__
    data_device = (float*) malloc(size_alloc);
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
    return column + (rows * row);
}

__host__ __device__ float& Matrix::Get(const int& index) {
    if (index >= rows * columns || index < 0) {
        printf("%s%d\n", "Warning: Out of bounds! Index: ", index);
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
        if (local) {
            std::cout << "Warning! Host locality is set to true, but device synchronization was called." <<
                "\n\nThis will likely result in a segfault, as the corresponding GPU table data was not initialized." << std::endl;
        }
    }
    else {
        ptr = device_alloc;
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
#ifndef __CUDA_ARCH__
    free(data);
#endif
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

    return matrix->DeviceTransfer(matrix->device_alloc, matrix);
}