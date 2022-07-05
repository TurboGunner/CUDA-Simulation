#include "fluid_sim_cuda.cuh"

__global__ void AddOnAxis(HashMap* data, IndexPair origin, float value, uint3 length) {
	data->Get(origin.IX(length.x)) += value;
}

cudaError_t AddOnAxisCuda(AxisData& current, const IndexPair& origin, const float& value, const uint3& length) {
	cudaError_t cuda_status = cudaSuccess;

	HashMap*& c_map = current.map_->device_alloc_;

	AddOnAxis<<<1, 1>>> (c_map, origin, value, length);

	cuda_status = PostExecutionChecks(cuda_status, "AddOnAxisKernel");
	return cuda_status;
}

__global__ void AddOnVector(HashMap* velocity_x, HashMap* velocity_y, HashMap* velocity_z, IndexPair origin, float3 value, uint3 length) {
	velocity_x->Get(origin.IX(length.x)) += value.x;
	velocity_y->Get(origin.IX(length.x)) += value.y;
	velocity_z->Get(origin.IX(length.x)) += value.z;
}

cudaError_t AddOnVectorCuda(VectorField& velocity, const IndexPair& origin, const float3& value, const uint3& length) {
	cudaError_t cuda_status = cudaSuccess;

	HashMap*& v_map_x = velocity.map_[0].map_->device_alloc_,
		*&v_map_y = velocity.map_[1].map_->device_alloc_,
		*&v_map_z = velocity.map_[2].map_->device_alloc_;

	AddOnVector<<<1, 1>>> (v_map_x, v_map_y, v_map_z, origin, value, length);

	cuda_status = PostExecutionChecks(cuda_status, "AddOnVectorKernel");
	return cuda_status;
}