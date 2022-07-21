#include "fluid_sim_cuda.cuh"
#include "lbm_sim_cuda.cuh"

__global__ void StreamKernel(HashMap* data, HashMap* data_prev, uint3 length) {
	unsigned int z_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;
	unsigned int x_bounds = blockIdx.z * blockDim.z + threadIdx.z + 1;

	IndexPair incident(x_bounds, y_bounds, z_bounds);

	//Primary Directions

	float left_value = data->Get(incident.Left().IX(length.x));
	data->Get(incident.Left().IX(length.x)) = data->Get(incident.Right().IX(length.x));
	data->Get(incident.Right().IX(length.x)) = left_value;

	float front_value = data->Get(incident.Front().IX(length.x));
	data->Get(incident.Front().IX(length.x)) = data->Get(incident.Back().IX(length.x));
	data->Get(incident.Back().IX(length.x)) = front_value;

	float up_value = data->Get(incident.Up().IX(length.x));
	data->Get(incident.Up().IX(length.x)) = data->Get(incident.Down().IX(length.x));
	data->Get(incident.Down().IX(length.x)) = up_value;

	//Up Corners

	float l_front_up_corner_value = data->Get(incident.CornerLUpFront().IX(length.x));
	data->Get(incident.CornerLUpFront().IX(length.x)) = data->Get(incident.CornerRDownBack().IX(length.x));
	data->Get(incident.CornerRDownBack().IX(length.x)) = l_front_up_corner_value;

	float l_back_up_corner_value = data->Get(incident.CornerLUpBack().IX(length.x));
	data->Get(incident.CornerLUpBack().IX(length.x)) = data->Get(incident.CornerRDownFront().IX(length.x));
	data->Get(incident.CornerRDownFront().IX(length.x)) = l_back_up_corner_value;

	//Down Corners

	float l_front_down_corner_value = data->Get(incident.CornerLDownFront().IX(length.x));
	data->Get(incident.CornerLDownFront().IX(length.x)) = data->Get(incident.CornerRUpBack().IX(length.x));
	data->Get(incident.CornerRUpBack().IX(length.x)) = l_front_down_corner_value;

	float l_back_down_corner_value = data->Get(incident.CornerLDownBack().IX(length.x));
	data->Get(incident.CornerLDownBack().IX(length.x)) = data->Get(incident.CornerRUpFront().IX(length.x));
	data->Get(incident.CornerRUpFront().IX(length.x)) = l_back_down_corner_value;

	//Mid Corners

	float l_mid_front_value = data->Get(incident.CornerLMidFront().IX(length.x));
	data->Get(incident.CornerLMidFront().IX(length.x)) = data->Get(incident.CornerRMidBack().IX(length.x));
	data->Get(incident.CornerRMidBack().IX(length.x)) = l_mid_front_value;

	float l_mid_back_value = data->Get(incident.CornerLMidBack().IX(length.x));
	data->Get(incident.CornerLMidBack().IX(length.x)) = data->Get(incident.CornerRMidFront().IX(length.x));
	data->Get(incident.CornerRMidFront().IX(length.x)) = l_mid_back_value;
}

cudaError_t StreamCuda(int bounds, AxisData& current, AxisData& previous, const uint3& length) {
	cudaError_t cuda_status = cudaSuccess;

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length.x);

	HashMap* c_map = current.map_->device_alloc_,
		* p_map = previous.map_->device_alloc_;

	StreamKernel<<<blocks, threads>>> (c_map, p_map, length);

	BoundaryConditionsCuda(bounds, current, length);

	cuda_status = PostExecutionChecks(cuda_status, "LBMStreamKernel");

	return cuda_status;
}