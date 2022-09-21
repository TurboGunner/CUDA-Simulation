#include "matrix.cuh"

__global__ void TransposeKernel(Matrix* matrix, Matrix* output) {
    unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_bounds <= output->rows && y_bounds <= output->columns) {
        output->Get(output->IX(threadIdx.y, threadIdx.x)) = matrix->Get(matrix->IX(threadIdx.x, threadIdx.y));
    }
}

__host__ __device__ Matrix* Matrix::TransposeGPU(Matrix* matrix) {
    cudaError_t cuda_status = cudaSuccess;

    Matrix* output = Matrix::Create(matrix->columns, matrix->rows);
#ifndef __CUDA__ARCH__
    cuda_status = output->DeviceTransfer(output->device_alloc, output);
#endif

    dim3 blocks, threads;

    blocks = dim3(1, 1);
    threads = dim3(matrix->columns, matrix->rows);

    TransposeKernel<<<blocks, threads>>> (matrix->device_alloc, output->device_alloc);
#ifndef __CUDA__ARCH__
    cuda_status = PostExecutionChecks(cuda_status, "TransposeKernel", true);
    output->HostTransfer();
#endif

    return output;
}

__global__ void MultiplyKernel(Matrix* matrix_A, Matrix* matrix_B, Matrix* output) {
    unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int idx = y_bounds * output->columns + x_bounds;

    if (y_bounds < output->rows && x_bounds < output->columns) {
        float Pvalue = 0;
        for (int k = 0; k < matrix_A->columns; ++k) {
            Pvalue += matrix_A->Get(y_bounds * matrix_A->columns + k) * matrix_B->Get(k * matrix_B->columns + x_bounds);
        }
        output->Set(idx, Pvalue);
    }
}

__host__ __device__ Matrix* Matrix::MultiplyGPU(Matrix* matrix_A, Matrix* matrix_B) {
    cudaError_t cuda_status = cudaSuccess;

    Matrix* output = Matrix::Create(matrix_A->rows, matrix_B->columns);
#ifndef __CUDA__ARCH__
    cuda_status = output->DeviceTransfer(output->device_alloc, output);
#endif

    const int TILE_DIM = 1;

    dim3 blocks, threads;
    threads = dim3(TILE_DIM, TILE_DIM);

    blocks = dim3(ceil((1.0 * output->columns) / TILE_DIM), ceil((1.0f * output->rows) / TILE_DIM), 1);

    MultiplyKernel<<<blocks, threads>>> (matrix_A->device_alloc, matrix_B->device_alloc, output->device_alloc);

#ifndef __CUDA__ARCH__
    cuda_status = PostExecutionChecks(cuda_status, "MultiplyKernel", true);
    output->HostTransfer();
#endif

    return output;
}