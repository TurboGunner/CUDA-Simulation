#include "fluid_sim_cuda.cuh"

__global__ void ProjectKernel(HashMap<IndexPair, F_Vector, Hash<IndexPair>>* velocity, HashMap<IndexPair, float, HashDupe<IndexPair>>* data, HashMap<IndexPair, float, HashDupe<IndexPair>>* data_prev, unsigned int length, unsigned int iter, int bounds) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		IndexPair incident(y_bounds, x_bounds);
		(*data)[incident] =
			((*velocity)[incident.Right()].vx_
				- (*velocity)[incident.Left()].vx_
				+ (*velocity)[incident.Up()].vy_
				- (*velocity)[incident.Down()].vy_
				* -0.5f) * (1.0f / length);
		(*data_prev)[IndexPair(y_bounds, x_bounds)] = 0;
	}
	if (x_bounds * y_bounds >= (length * length)) {
		BoundaryConditions(0, data, length);
		BoundaryConditions(0, data_prev, length);
		LinearSolverGPU(data, data_prev, 1, 4, length, iter, bounds);
	}

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		IndexPair incident(y_bounds, x_bounds);
		(*velocity)[incident].vx_ -= 0.5f
			* ((*data_prev)[incident.Right()]
			- (*data_prev)[incident.Left()])
			* length;
		(*velocity)[incident].vy_ -= 0.5f
			* ((*data_prev)[incident.Up()]
			- (*data_prev)[incident.Down()])
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

	AxisData v_prev_x(length, Axis::X), v_prev_y(length, Axis::Y);

	velocity_prev.DataConstrained(Axis::X, v_prev_x);
	velocity_prev.DataConstrained(Axis::Y, v_prev_y);

	ProjectKernel<<<blocks, threads>>> (velocity.GetVectorMap(), v_prev_x.map_, v_prev_y.map_, length, iter, bounds);

	handler.PostExecutionChecks(cuda_status);
}