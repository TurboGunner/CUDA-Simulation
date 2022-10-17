#include "mpm.cuh"

__global__ void UpdateGrid(Grid* grid) {
	//Cell Boundaries
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds); //Current Index

	grid->GetCellVelocity(incident) /= grid->GetCellMass(incident); //Converting momentum to velocity

	//Applying gravity to velocity
	Vector3D gravity_vector(0.0f, 0.0f, grid->gravity);
	grid->GetCellVelocity(incident) += (gravity_vector * grid->dt);

	//Boundary Conditions
	if (x_bounds < 2 || x_bounds > grid->side_size_ - 3) {
		grid->GetCellVelocity(incident).dim[0] = 0;
	}
	if (y_bounds < 2 || y_bounds > grid->side_size_ - 3) {
		grid->GetCellVelocity(incident).dim[1] = 0;
	}
	if (z_bounds < 2 || z_bounds > grid->side_size_ - 3) {
		grid->GetCellVelocity(incident).dim[2] = 0;
	}
}

__device__ Vector3D* GetWeights(Vector3D cell_difference) { //Returns weights shared
	Vector3D weights[3]{}; //Array of weights

	weights[0] = (cell_difference.Negative() + 0.5f).Squared() * 0.5f;
	weights[1] = cell_difference.Squared().Negative() + 0.75f;
	weights[2] = (cell_difference + 0.5f).Squared() * 0.5f;

	return weights;
}

__global__ void ClearGrid(Grid* grid) {
	//Cell Boundaries
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds); //Current position

	grid->GetCellMass(incident) = 0;
	grid->GetCellVelocity(incident).Reset();
}

__host__ void Grid::CalculateBounds() {
	unsigned int threads_per_dim = (unsigned int)cbrt(1);
	unsigned int block_count = ((side_size_ + threads_per_dim) - 1) / (threads_per_dim);

	cell_threads = dim3(threads_per_dim, threads_per_dim, threads_per_dim);
	cell_blocks = dim3(block_count, block_count, block_count);

	std::cout << "Resolution: " << GetResolution() << std::endl;

	particle_threads = dim3(threads_per_dim * GetResolution(), threads_per_dim, threads_per_dim); //NOTE
	particle_blocks = dim3(block_count, block_count, block_count);
}

__global__ inline static void TestInitKernel(Grid* grid) {
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds);

	grid->GetParticleMass(incident) = 1.0f;
}

__host__ cudaError_t Grid::SimulateGPU(Grid* grid, cudaStream_t& cuda_stream) {
	cudaError_t cuda_status = cudaSuccess;

	dim3& blocks = grid->cell_blocks, & threads = grid->cell_threads;
	dim3& blocks2 = grid->particle_blocks, & threads2 = grid->particle_threads;

	if (!grid->up_to_date_) {
		grid->CalculateBounds();
		grid->DeviceTransfer(grid);

		TestInitKernel<<<blocks2, threads2, 0, cuda_stream>>> (grid);
		cudaStreamSynchronize(cuda_stream);
	}

	ClearGrid<<<blocks, threads, 0, cuda_stream>>> (grid);
	cuda_status = PostExecutionChecks(cuda_status, "ClearGrid", false);
	CudaExceptionHandler(cuda_status, "ClearGrid failed!");

	//std::cout << "Allocated successfully " << grid->GetTotalSize() << " cells! (host)" << std::endl;
	//std::cout << "Allocated successfully " << grid->GetParticleCount() << " particles! (host)" << std::endl;

	UpdateCell<<<blocks2, threads2, 0, cuda_stream>>> (grid, grid->momentum_matrix->device_alloc, grid->cell_dist_matrix->device_alloc, grid->momentum->device_alloc);
	cuda_status = PostExecutionChecks(cuda_status, "CellMomentum", false);
	CudaExceptionHandler(cuda_status, "CellMomentum failed!");
	//cudaStreamSynchronize(cuda_stream);

	//std::cout << "Ran through cell momentum!" << std::endl;

	SimulateGrid<<<blocks2, threads2, 0, cuda_stream>>> (grid, grid->stress_matrix->device_alloc, grid->weighted_stress->device_alloc, grid->cell_dist_matrix->device_alloc, grid->momentum->device_alloc, grid->viscosity_term->device_alloc); //NOTE
	cuda_status = PostExecutionChecks(cuda_status, "VelocityGradientSolve", false);
	CudaExceptionHandler(cuda_status, "SimulateGrid failed!");

	//std::cout << "Ran through the velocity gradient solve!" << std::endl;

	UpdateGrid<<<blocks, threads, 0, cuda_stream>>> (grid);
	cuda_status = PostExecutionChecks(cuda_status, "UpdateGrid", false);
	CudaExceptionHandler(cuda_status, "UpdateGrid failed!");

	//std::cout << "Updated the grid!" << std::endl;

	AdvectParticles<<<blocks2, threads2, 0, cuda_stream>>> (grid, grid->B_term->device_alloc, grid->weighted_term->device_alloc);
	cuda_status = PostExecutionChecks(cuda_status, "AdvectParticles", false);
	CudaExceptionHandler(cuda_status, "AdvectParticles failed!");

	//cudaStreamSynchronize(cuda_stream);

	//std::cout << "Advected the particles!" << std::endl;

	//cuda_status = grid->HostTransfer();

	return cuda_status;
}