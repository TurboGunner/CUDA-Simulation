#include "fluid_sim_cuda.cuh"

__global__ void ProjectKernel(float3* velocity, float* data, float* data_prev, unsigned int length, unsigned int iter, int bounds) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		data[IX(x_bounds, y_bounds + 1, length)] =
			((velocity[IX(x_bounds + 1, y_bounds + 1, length)].x
				- velocity[IX(x_bounds - 1, y_bounds + 1, length)].x
				+ velocity[IX(x_bounds, y_bounds + 2, length)].y
				- velocity[IX(x_bounds, y_bounds, length)].y)
				* -0.5f) * (1.0f / length);
		//data_prev[IX(x_bounds, y_bounds + 1, length)] = 0;
	}
	if (x_bounds * y_bounds >= (length * length)) {
		PointerBoundaries(data, length);
		PointerBoundaries(data_prev, length);
		LinearSolverGPU(data, data_prev, 1, 4, length, iter, bounds);
	}

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		velocity[IX(x_bounds, y_bounds + 1, length)].x -= 0.5f
			* (data_prev[IX(x_bounds + 1, y_bounds + 1, length)]
			- data_prev[IX(x_bounds - 1, y_bounds + 1, length)])
			* length;
		velocity[IX(x_bounds, y_bounds + 1, length)].y -= 0.5f
			* (data_prev[IX(x_bounds, y_bounds + 2, length)]
			- data_prev[IX(x_bounds, y_bounds, length)])
			* length;
	}
	if (x_bounds * y_bounds >= (length * length)) {
		if (bounds == 0) {
			PointerBoundariesVector(velocity, length);
		}
		else {
			PointerBoundariesSpecial(velocity, length);
		}
	}
}

void ProjectCuda(int bounds, VectorField& velocity, VectorField& velocity_prev, const unsigned int& length, const unsigned int& iter) {
	unsigned int alloc_size = length * length;
	CudaMethodHandler handler(alloc_size, "ProjectCudaKernel");

	float3* v_ptr = velocity.FlattenMap(), * v_copy_ptr = nullptr;
	float* v_x_prev_ptr = velocity_prev.FlattenMapX(), * v_x_prev_copy_ptr = nullptr;
	float* v_y_prev_ptr = velocity_prev.FlattenMapY(), * v_y_prev_copy_ptr = nullptr;

	handler.float_copy_ptrs_.insert(handler.float_copy_ptrs_.end(), { v_x_prev_copy_ptr, v_y_prev_copy_ptr });
	handler.float_ptrs_.insert(handler.float_ptrs_.end(), { v_x_prev_ptr, v_y_prev_ptr });

	handler.float3_copy_ptrs_.insert(handler.float3_copy_ptrs_.end(), { v_copy_ptr });
	handler.float3_ptrs_.insert(handler.float3_ptrs_.end(), { v_ptr });

	handler.AllocateCopyPointers();

	cudaError_t cuda_status = cudaSuccess;

	cuda_status = handler.CopyToMemory(cudaMemcpyHostToDevice, cuda_status);

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length);

	ProjectKernel<<<blocks, threads>>> (v_copy_ptr, v_x_prev_copy_ptr, v_y_prev_copy_ptr, length, iter, bounds);

	handler.PostExecutionChecks(cuda_status);

	cuda_status = CopyFunction("cudaMemcpy failed at v_x_ptr!", v_x_prev_ptr, v_x_prev_copy_ptr,
		cudaMemcpyDeviceToHost, cuda_status, (size_t)alloc_size,
		sizeof(float));

	cuda_status = CopyFunction("cudaMemcpy failed at v_y_ptr!", v_y_prev_ptr, v_y_prev_copy_ptr,
		cudaMemcpyDeviceToHost, cuda_status, (size_t)alloc_size,
		sizeof(float));

	float3* ptr = new float3[alloc_size];

	cuda_status = CopyFunction("cudaMemcpy failed at v_ptr!", ptr, v_copy_ptr,
		cudaMemcpyDeviceToHost, cuda_status, (size_t)alloc_size,
		sizeof(float3));

	velocity.RepackMapVector(ptr);
	velocity_prev.RepackMap(v_x_prev_ptr, v_y_prev_ptr);

	handler.float3_ptrs_.insert(handler.float3_ptrs_.end(), { ptr });
	std::cout << v_y_prev_ptr[1] << std::endl;
	handler.~CudaMethodHandler();
	std::cout << velocity_prev.GetVectorMap()[IndexPair(1, 1)].ToString() << std::endl;
}