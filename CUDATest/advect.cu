#include "fluid_sim_cuda.cuh"

__device__ inline void AveragerLogic(float& value, float& current, float& previous, const unsigned int& length) {
	if (value < 0.5f) {
		value = 0.5f;
	}
	if (value > length + 0.5f) {
		value = length + 0.5f;
	}
	current = floorf(value);
	previous = current + 1.0f;
}

__global__ void AdvectKernel(HashMap* data, HashMap* data_prev, HashMap* velocity_x, HashMap* velocity_y, HashMap* velocity_z, float dt, uint3 length, int bounds) {
	unsigned int z_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;
	unsigned int x_bounds = blockIdx.z * blockDim.z + threadIdx.z + 1;

	float x_current, x_previous,
		y_current, y_previous,
		z_current, z_previous;

	float x_dt = dt * (length.x - 2),
		y_dt = dt * (length.y - 2),
		z_dt = dt * (length.z - 2);

	float x_fac_current, x_fac_prev,
		y_fac_current, y_fac_prev,
		z_fac_current, z_fac_prev;

	float x_value, y_value, z_value;

	x_value = x_bounds - (x_dt * velocity_x->Get(IndexPair(x_bounds, y_bounds, z_bounds).IX(length.x)));
	y_value = y_bounds - (y_dt * velocity_y->Get(IndexPair(x_bounds, y_bounds, z_bounds).IX(length.x)));
	z_value = z_bounds - (z_dt * velocity_z->Get(IndexPair(x_bounds, y_bounds, z_bounds).IX(length.x)));
	//printf("%f\n", (x_dt * velocity_x->Get(IndexPair(x_bounds, y_bounds, z_bounds).IX(length.x))));
	//printf("%f\n", (y_dt * velocity_y->Get(IndexPair(x_bounds, y_bounds, z_bounds).IX(length.x))));
	//printf("%f\n", (z_dt * velocity_z->Get(IndexPair(x_bounds, y_bounds, z_bounds).IX(length.x))));

	AveragerLogic(x_value, x_current, x_previous, length.x);
	AveragerLogic(y_value, y_current, y_previous, length.y);
	AveragerLogic(z_value, z_current, z_previous, length.z);

	x_fac_prev = x_value - x_current;
	x_fac_current = 1.0f - x_fac_prev;
	y_fac_prev = y_value - y_current;
	y_fac_current = 1.0f - y_fac_prev;
	z_fac_prev = z_value - z_current;
	z_fac_current = 1.0f - z_fac_prev;

	float compute_x_current_1 = y_fac_current *
		(data->Get(IndexPair(x_current, y_current, z_current).IX(length.x)) * z_fac_current)
		+ (data->Get(IndexPair(x_current, y_current, z_previous).IX(length.x)) * z_fac_prev);

	float compute_x_current_2  = y_fac_prev *
		(data->Get(IndexPair(x_current, y_previous, z_current).IX(length.x)) * z_fac_current)
		+ (data->Get(IndexPair(x_current, y_previous, z_previous).IX(length.x)) * z_fac_prev);

	float compute_x_current = x_fac_current * (compute_x_current_1 + compute_x_current_2);

	float compute_x_prev_1 = y_fac_current *
		(data->Get(IndexPair(x_previous, y_current, z_current).IX(length.x)) * z_fac_current)
		+ (data->Get(IndexPair(x_previous, y_current, z_previous).IX(length.x)) * z_fac_prev);

	float compute_x_prev_2 = y_fac_prev *
		(data->Get(IndexPair(x_previous, y_previous, z_current).IX(length.x)) * z_fac_current)
		+ (data->Get(IndexPair(x_previous, y_previous, z_previous).IX(length.x)) * z_fac_prev);

	float compute_x_prev = x_fac_prev * (compute_x_prev_1 + compute_x_prev_2);

	data->Put(IndexPair(x_bounds, y_bounds, z_bounds).IX(length.x), (compute_x_current + compute_x_prev));
}

cudaError_t AdvectCuda(int bounds, AxisData& current, AxisData& previous, VectorField& velocity, const float& dt, const uint3& length) {
	cudaError_t cuda_status = cudaSuccess;

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length.x);

	HashMap*& v_map_x = velocity.map_[0].map_->device_alloc_,
		*&v_map_y = velocity.map_[1].map_->device_alloc_,
		*&v_map_z = velocity.map_[2].map_->device_alloc_,
		*&c_map = current.map_->device_alloc_,
		*&p_map = previous.map_->device_alloc_;

	std::cout << "bidoof" << std::endl;

	AdvectKernel<<<blocks, threads>>> (c_map, p_map, v_map_x, v_map_y, v_map_z, dt, length, bounds);
	BoundaryConditionsCuda(bounds, current, length);

	cuda_status = PostExecutionChecks(cuda_status, "AdvectCudaKernel");
	cudaDeviceSynchronize();
	return cuda_status;
}