#include "fluid_sim_cuda.cuh"

__global__ void LinearSolverKernel(HashMap<IndexPair, float, Hash>* data, HashMap<IndexPair, float, Hash>* data_prev, float a_fac, float c_fac, unsigned int length, unsigned int iter, int bounds) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	auto* pairs = GetAdjacentCoordinates(IndexPair(y_bounds, x_bounds), data->size_);

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		for (int i = 0; i < iter; i++) {
			data[pairs->Get(FluidSim::Direction::Origin)] = ((*data_prev)[pairs->Get(FluidSim::Direction::Origin)] +
				a_fac *
				(*data)[pairs->Get(FluidSim::Direction::Right)]
					+ (*data)[pairs->Get(FluidSim::Direction::Left)]
					+ (*data)[pairs->Get(FluidSim::Direction::Up)]
					+ (*data)[pairs->Get(FluidSim::Direction::Down)])
				* (1.0f / c_fac);
		}
		if (x_bounds * y_bounds >= (length * length)) {
			if (bounds == 0) {
				PointerBoundaries(data, length);
			}
			if (bounds == 1) {
				PointerBoundariesSpecialX(data, length);
			}
			else {
				PointerBoundariesSpecialY(data, length);
			}
		}
	}
}

void LinearSolverCuda(int bounds, AxisData& current, AxisData& previous, const float& a_fac, const float& c_fac, const unsigned int& iter, const unsigned int& length) {
	unsigned int alloc_size = length * length;
	CudaMethodHandler handler(alloc_size, "LinearSolverKernel");

	cudaError_t cuda_status = cudaSuccess;

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length);

	LinearSolverKernel<<<blocks, threads>>> (current.map_, previous.map_, a_fac, c_fac, length, iter, bounds);

	handler.PostExecutionChecks(cuda_status);
}