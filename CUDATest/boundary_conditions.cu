#include "fluid_sim_cuda.cuh"

__global__ void BoundaryConditions(int bounds, HashMap* c_map, uint3 size) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;
	unsigned int length = size.x;
	unsigned int bound = length - 1;

	c_map->Get(IndexPair(x_bounds, y_bounds, 0).IX(length)) = bounds == 3 ? -c_map->Get(IndexPair(x_bounds, y_bounds, 1).IX(length)) : c_map->Get(IndexPair(x_bounds, y_bounds, 1).IX(length));
	c_map->Get(IndexPair(x_bounds, y_bounds, bound).IX(length)) = bounds == 3 ? -c_map->Get(IndexPair(x_bounds, y_bounds, bound - 1).IX(length)) : c_map->Get(IndexPair(x_bounds, y_bounds, bound - 1).IX(length));

	c_map->Get(IndexPair(y_bounds, 0, x_bounds).IX(length)) = bounds == 2 ? -c_map->Get(IndexPair(x_bounds, 1, y_bounds).IX(length)) : c_map->Get(IndexPair(x_bounds, 1, y_bounds).IX(length));
	c_map->Get(IndexPair(y_bounds, bound, x_bounds).IX(length)) = bounds == 2 ? -c_map->Get(IndexPair(x_bounds, bound - 1, y_bounds).IX(length)) : c_map->Get(IndexPair(x_bounds, bound - 1, y_bounds).IX(length));

	c_map->Get(IndexPair(0, y_bounds, x_bounds).IX(length)) = bounds == 1 ? -c_map->Get(IndexPair(1, x_bounds, y_bounds).IX(length)) : c_map->Get(IndexPair(1, x_bounds, y_bounds).IX(length));
	c_map->Get(IndexPair(bound, y_bounds, x_bounds).IX(length)) = bounds == 1 ? -c_map->Get(IndexPair(bound - 1, x_bounds, y_bounds).IX(length)) : c_map->Get(IndexPair(bound - 1, x_bounds, y_bounds).IX(length));

	if (x_bounds == size.x - 4 && y_bounds == size.y - 4) { //Correction for check due to bounded thread changes
		c_map->Get(IndexPair(0, 0, 0).IX(length)) = .33f * //Min X, Min Y, Min Z
			(c_map->Get(IndexPair(1, 0, 0).IX(length))
			+ c_map->Get(IndexPair(0, 1, 0).IX(length))
			+ c_map->Get(IndexPair(0, 0, 1).IX(length)));

		c_map->Get(IndexPair(0, bound, 0).IX(length)) = .33f * //Min X, Max Y, Min Z
			(c_map->Get(IndexPair(1, bound, 0).IX(length))
			+ c_map->Get(IndexPair(0, bound - 1, 0).IX(length))
			+ c_map->Get(IndexPair(0, bound, 1).IX(length)));

		c_map->Get(IndexPair(0, 0, bound).IX(length)) = .33f * //Min X, Min Y, Max Z
			(c_map->Get(IndexPair(1, 0, bound).IX(length))
			+ c_map->Get(IndexPair(0, 1, bound).IX(length))
			+ c_map->Get(IndexPair(0, 0, length).IX(length)));

		c_map->Get(IndexPair(bound, 0, 0).IX(length)) = .33f * //Max X, Min Y, Min Z
			(c_map->Get(IndexPair(bound - 1, 0, 0).IX(length))
			+ c_map->Get(IndexPair(bound, 1, 0).IX(length))
			+ c_map->Get(IndexPair(bound, 0, 1).IX(length)));

		c_map->Get(IndexPair(bound, bound, 0).IX(length)) = .33f * //Max X, Max Y, Min Z
			(c_map->Get(IndexPair(bound - 1, bound, 0).IX(length))
			+ c_map->Get(IndexPair(bound, bound - 1, 0).IX(length))
			+ c_map->Get(IndexPair(bound, bound, 1).IX(length)));

		c_map->Get(IndexPair(bound, 0, bound).IX(length)) = .33f * //Max X, Min Y, Max Z
			(c_map->Get(IndexPair(bound - 1, 0, bound).IX(length))
			+ c_map->Get(IndexPair(bound, 1, bound).IX(length))
			+ c_map->Get(IndexPair(bound, 0, bound - 1).IX(length)));


		c_map->Get(IndexPair(0, bound, bound).IX(length)) = .33f * //Max X, Min Y, Max Z
			(c_map->Get(IndexPair(1, bound, bound).IX(length))
			+ c_map->Get(IndexPair(0, bound - 1, bound).IX(length))
			+ c_map->Get(IndexPair(0, bound, bound - 1).IX(length)));

		c_map->Get(IndexPair(bound, bound, bound).IX(length)) = .33f * //Max X, Max Y, Max Z
			(c_map->Get(IndexPair(bound - 1, bound, bound).IX(length))
			+ c_map->Get(IndexPair(bound, bound - 1, bound).IX(length))
			+ c_map->Get(IndexPair(bound, bound, bound - 1).IX(length)));
	}
}

cudaError_t BoundaryConditionsCuda(int bounds, HashMap* map, const uint3& length) {
	cudaError_t cuda_status = cudaSuccess;

	dim3 blocks, threads;
	ThreadAllocator2D(blocks, threads, length.x);

	BoundaryConditions<<<blocks, threads>>> (bounds, map, length);
	cuda_status = PostExecutionChecks(cuda_status, "BoundaryConditionsKernel");
	return cuda_status;
}