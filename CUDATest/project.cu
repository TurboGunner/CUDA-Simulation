#include "fluid_sim_cuda.cuh"

__global__ void ProjectKernel(HashMap<F_Vector>* velocity, HashMap<float>* data, HashMap<float>* data_prev, unsigned int length, int bounds) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) { 
		IndexPair incident(y_bounds, x_bounds);
		data->Get(incident.IX(length)) =
			((velocity->Get(incident.Right().IX(length)).vx_
				- velocity->Get(incident.Left().IX(length)).vx_
				+ velocity->Get(incident.Up().IX(length)).vy_
				- velocity->Get(incident.Down().IX(length)).vy_)
				* -0.5f) / length;
		data_prev->Get(incident.IX(length)) = 0;
	}
	if (x_bounds * y_bounds >= (length * length)) {
		BoundaryConditions(0, data, length);
		BoundaryConditions(0, data_prev, length);
	}
}

__global__ void ProjectKernel2(HashMap<F_Vector>* velocity, HashMap<F_Vector>* velocity_output, HashMap<float>* data, HashMap<float>* data_prev, unsigned int length, int bounds) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if (threadIdx.x < length - 1 && threadIdx.y < length - 1) {
		IndexPair incident(y_bounds, x_bounds);
		float compute_x = velocity->Get(incident.IX(length)).vx_ - (-0.5f *
			(data_prev->Get(incident.Right().IX(length))
				- data_prev->Get(incident.Left().IX(length)))
			* length);

		float compute_y = velocity->Get(incident.IX(length)).vy_ - (-0.5f *
			(data_prev->Get(incident.Up().IX(length))
				- data_prev->Get(incident.Down().IX(length)))
			* length);
		velocity_output->Put(incident.IX(length), F_Vector(compute_x, compute_y));
	}
	if (x_bounds * y_bounds >= (length * length)) {
		BoundaryConditions(bounds, velocity_output, length);
	}
}

void ProjectCuda(int bounds, VectorField& velocity, VectorField& velocity_prev, const unsigned int& length, const unsigned int& iter) {
	unsigned int alloc_size = length * length;

	cudaError_t cuda_status = cudaSuccess;

	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, length);

	AxisData v_prev_x(length, Axis::X), v_prev_y(length, Axis::Y);

	velocity_prev.DataConstrained(Axis::X, v_prev_x);
	velocity_prev.DataConstrained(Axis::Y, v_prev_y);

	HashMap<F_Vector>* v_map = nullptr, *v_output = nullptr;
	HashMap<float>* x_map = nullptr, *y_map = nullptr;
	velocity.GetVectorMap()->DeviceTransfer(cuda_status, velocity.GetVectorMap(), v_map);
	v_prev_x.map_->DeviceTransfer(cuda_status, v_prev_x.map_, x_map);
	v_prev_y.map_->DeviceTransfer(cuda_status, v_prev_y.map_, y_map);

	HashMap<F_Vector>* v_temp_output = new HashMap<F_Vector>(alloc_size);
	v_temp_output->DeviceTransfer(cuda_status, v_temp_output, v_output);

	ProjectKernel<<<blocks, threads>>> (v_map, y_map, x_map, length, bounds);
	LinearSolverKernel<<<blocks, threads>>> (y_map, x_map, 1, 4, length, iter, bounds);
	ProjectKernel2<<<blocks, threads>>> (v_map, v_output, y_map, x_map, length, bounds);

	std::cout << "Yo Pierre, you wanna come out here? *door squeaking noise*" << std::endl;

	PostExecutionChecks(cuda_status, "ProjectCudaKernel");

	v_temp_output->HostTransfer(cuda_status);
	v_prev_x.map_->HostTransfer(cuda_status);
	v_prev_y.map_->HostTransfer(cuda_status);
	velocity.GetVectorMap() = v_temp_output;

	velocity_prev.RepackFromConstrained(v_prev_x);
	velocity_prev.RepackFromConstrained(v_prev_y);
}