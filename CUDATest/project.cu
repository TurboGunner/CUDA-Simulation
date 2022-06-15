#include "fluid_sim_cuda.cuh"

__global__ void ProjectKernel(HashMap<IndexPair, F_Vector, Hash>* velocity, HashMap<IndexPair, float, Hash>* data, HashMap<IndexPair, float, Hash>* data_prev, unsigned int length, unsigned int iter, int bounds) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		auto* pairs = GetAdjacentCoordinates(IndexPair(y_bounds, x_bounds), velocity->size_);
		(*data)[pairs->Get(FluidSim::Direction::Origin)] =
			((*velocity)[pairs->Get(FluidSim::Direction::Right)].vx_
				- (*velocity)[pairs->Get(FluidSim::Direction::Left)].vx_
				+ (*velocity)[pairs->Get(FluidSim::Direction::Up)].vy_
				- (*velocity)[pairs->Get(FluidSim::Direction::Down)].vy_
				* -0.5f) * (1.0f / length);
		(*data_prev)[IndexPair(y_bounds, x_bounds)] = 0;
	}
	if (x_bounds * y_bounds >= (length * length)) {
		PointerBoundaries(data, length);
		PointerBoundaries(data_prev, length);
		LinearSolverGPU(data, data_prev, 1, 4, length, iter, bounds);
	}

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		auto* pairs = GetAdjacentCoordinates(IndexPair(y_bounds, x_bounds), velocity->size_);
		(*velocity)[pairs->Get(FluidSim::Direction::Origin)].vx_ -= 0.5f
			* ((*data_prev)[pairs->Get(FluidSim::Direction::Right)]
			- (*data_prev)[pairs->Get(FluidSim::Direction::Left)])
			* length;
		(*velocity)[pairs->Get(FluidSim::Direction::Origin)].vy_ -= 0.5f
			* ((*data_prev)[pairs->Get(FluidSim::Direction::Up)]
			- (*data_prev)[pairs->Get(FluidSim::Direction::Down)])
			* length;
	}
	if (x_bounds * y_bounds >= (length * length)) {
		BoundaryConditions(bounds, velocity, length);
	}
}

void ProjectCuda(int bounds, VectorField& velocity, VectorField& velocity_prev, const unsigned int& length, const unsigned int& iter) {
	unsigned int alloc_size = length * length;
	CudaMethodHandler handler(alloc_size, "ProjectCudaKernel");

	cudaError_t cuda_status = cudaSuccess;

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length);

	AxisData v_prev_x, v_prev_y;

	velocity_prev.DataConstrained(Axis::X, v_prev_x);
	velocity_prev.DataConstrained(Axis::Y, v_prev_y);

	ProjectKernel<<<blocks, threads>>> (velocity.map_, v_prev_x.map_, v_prev_y.map_, length, iter, bounds);

	handler.PostExecutionChecks(cuda_status);
}