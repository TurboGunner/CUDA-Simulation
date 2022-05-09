#include "fluid_sim_cuda.cuh"
#include "handler_methods.hpp"
#include "fluid_sim.hpp"

__global__ void LinearSolverKernel(float* data, const float* data_prev, float a_fac, float c_fac, unsigned int length, unsigned int iter, int bounds) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		for (int i = 0; i < iter; i++) {
			data[IX(x_bounds, y_bounds + 1, length)] = (data_prev[IX(x_bounds, y_bounds + 1, length)] +
				a_fac *
				(data[IX(x_bounds + 1, y_bounds + 1, length)]
					+ data[IX(x_bounds - 1, y_bounds + 1, length)]
					+ data[IX(x_bounds, y_bounds + 2, length)]
					+ data[IX(x_bounds, y_bounds, length)]))
				* (1.0f / c_fac);
		}
		if (x_bounds * y_bounds >= (length * length)) {
			if (bounds == 0) {
				PointerBoundaries(data, length);
			}
			else {
				PointerBoundariesSpecialX(data, length);
			}
		}
	}
}

void LinearSolverCuda(int bounds, VectorField& current, VectorField& previous, const float& a_fac, const float& c_fac, const unsigned int& iter, const unsigned int& length) {
	unsigned int alloc_size = length * length;
	CudaMethodHandler handler(alloc_size, "LinearSolverKernel");

	float* curr_copy_ptr = nullptr, * prev_copy_ptr = nullptr;

	float* current_ptr = current.FlattenMapX(),
		* prev_ptr = previous.FlattenMapX();

	handler.float_copy_ptrs_.insert(handler.float_copy_ptrs_.end(), { curr_copy_ptr, prev_copy_ptr });
	handler.float_ptrs_.insert(handler.float_ptrs_.end(), { current_ptr, prev_ptr });

	handler.AllocateCopyPointers();

	cudaError_t cuda_status = cudaSuccess;

	cuda_status = handler.CopyToMemory(cudaMemcpyHostToDevice, cuda_status);

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length);

	LinearSolverKernel<<<blocks, threads>>> (curr_copy_ptr, prev_copy_ptr, a_fac, c_fac, length, iter, bounds);

	handler.PostExecutionChecks(cuda_status);

	cuda_status = CopyFunction("cudaMemcpy failed!", current_ptr, curr_copy_ptr,
		cudaMemcpyDeviceToHost, cuda_status, (size_t)alloc_size,
		sizeof(float));

	current.RepackMap(current_ptr, current_ptr);
	handler.~CudaMethodHandler();
}