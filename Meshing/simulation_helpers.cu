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

__host__ cudaError_t Grid::SimulateGPU(Grid* grid, cudaStream_t& cuda_stream) {
	//Matrix Allocations
	cudaError_t cuda_status = cudaSuccess;

	Matrix* cell_dist_matrix = Matrix::Create(3, 1, false);
	Matrix* momentum = Matrix::Create(3, 1, false);
	Matrix* momentum_matrix = Matrix::Create(3, 3, false);

	Matrix* stress_matrix = Matrix::Create(3, 3, false);
	Matrix* weighted_stress = Matrix::Create(3, 3, false);

	Matrix* B_term = Matrix::Create(3, 3, false);
	Matrix* weighted_term = Matrix::Create(3, 3, false);

	Matrix* viscosity_term = Matrix::Create(3, 3, false);

	grid->DeviceTransfer(grid);

	std::cout << "Allocated successfully " << grid->GetTotalSize() << " cells! (host)" << std::endl;
	std::cout << "Allocated successfully " << grid->GetParticleCount() << " particles! (host)" << std::endl;

	dim3 blocks, threads;
	unsigned int threads_per_dim = (unsigned int) cbrt (1);
	unsigned int block_count = ((grid->side_size_ + threads_per_dim) - 1) / (threads_per_dim);

	threads = dim3(threads_per_dim, threads_per_dim, threads_per_dim);
	blocks = dim3(block_count, block_count, block_count);
	dim3 blocks2, threads2;

	std::cout << "Resolution: " << grid->GetResolution() << std::endl;

	threads2 = dim3(threads_per_dim * grid->GetResolution(), threads_per_dim, threads_per_dim); //NOTE
	blocks2 = dim3(block_count, block_count, block_count);

	UpdateCell<<<blocks2, threads2, 0, cuda_stream>>> (grid->device_alloc_, momentum_matrix->device_alloc, cell_dist_matrix->device_alloc, momentum->device_alloc);
	cuda_status = PostExecutionChecks(cuda_status, "CellMomentum", false);

	//cuda_status = cudaDeviceSynchronize();
	CudaExceptionHandler(cuda_status, "CellMomentum failed!");

	std::cout << "Ran through cell momentum!" << std::endl;

	SimulateGrid<<<blocks2, threads2, 0, cuda_stream>>> (grid->device_alloc_, stress_matrix->device_alloc, weighted_stress->device_alloc, cell_dist_matrix->device_alloc, momentum->device_alloc, viscosity_term); //NOTE
	cuda_status = PostExecutionChecks(cuda_status, "VelocityGradientSolve", false);

	//cuda_status = cudaDeviceSynchronize();
	CudaExceptionHandler(cuda_status, "SimulateGrid failed!");

	std::cout << "Ran through the velocity gradient solve!" << std::endl;

	UpdateGrid<<<blocks, threads, 0, cuda_stream>>> (grid->device_alloc_);
	cuda_status = PostExecutionChecks(cuda_status, "UpdateGrid", false);

	//cuda_status = cudaDeviceSynchronize();
	CudaExceptionHandler(cuda_status, "UpdateGrid failed!");

	std::cout << "Updated the grid!" << std::endl;

	AdvectParticles<<<blocks2, threads2, 0, cuda_stream>>> (grid->device_alloc_, B_term, weighted_term);
	cuda_status = PostExecutionChecks(cuda_status, "AdvectParticles", false);

	//cuda_status = cudaDeviceSynchronize();
	CudaExceptionHandler(cuda_status, "AdvectParticles failed!");

	std::cout << "Advected the particles!" << std::endl;

	cuda_status = grid->HostTransfer();

	return cuda_status;
}