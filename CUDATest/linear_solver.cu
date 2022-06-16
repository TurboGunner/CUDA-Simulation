#include "fluid_sim_cuda.cuh"

__global__ void LinearSolverKernel(HashMap<IndexPair, float, HashDupe<IndexPair>>* data, HashMap<IndexPair, float, HashDupe<IndexPair>>* data_prev, float a_fac, float c_fac, unsigned int length, unsigned int iter, int bounds) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y;

	//printf("%u %u \n", y_bounds, x_bounds);
	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		auto* pairs = GetAdjacentCoordinates(IndexPair(y_bounds, x_bounds));
		printf("%d", pairs->size_);
		for (int i = 0; i < iter; i++) {
			printf("%f", pairs->Get(Direction::Origin).x);
			(*data)[pairs->Get(Direction::Origin)] = ((*data_prev)[pairs->Get(Direction::Origin)] +
				a_fac *
				(*data)[pairs->Get(Direction::Right)]
					+ (*data)[pairs->Get(Direction::Left)]
					+ (*data)[pairs->Get(Direction::Up)]
					+ (*data)[pairs->Get(Direction::Down)])
				* (1.0f / c_fac);
		}
		if (x_bounds * y_bounds >= (length * length)) {
			BoundaryConditions(bounds, data, length);
		}
	}
}

void LinearSolverCuda(int bounds, AxisData& current, AxisData& previous, const float& a_fac, const float& c_fac, const unsigned int& iter, const unsigned int& length) {
	unsigned int alloc_size = length * length;
	CudaMethodHandler handler(alloc_size, "LinearSolverKernel");

	cudaError_t cuda_status = cudaSuccess;

	dim3 blocks, threads;
	std::cout << "Length: " << length << std::endl;
	ThreadAllocator(blocks, threads, length);

	LinearSolverKernel<<<blocks, threads>>> (current.map_, previous.map_, a_fac, c_fac, length, iter, bounds);

	handler.PostExecutionChecks(cuda_status);
}