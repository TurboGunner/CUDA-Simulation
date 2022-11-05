#include "mpm.cuh"

inline void CurandCallCheck(const curandStatus_t& status) {
	if (status != CURAND_STATUS_SUCCESS) {
		ProgramLog::OutputLine("Warning! CUDA Rand Status is not successful! State: " + std::to_string(status));
	}
}

__global__ void UpdateGrid(Grid* grid) {
	//Cell Boundaries
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds); //Current Index
	if (grid->GetCellMass(incident) > 0) {
		grid->GetCellVelocity(incident) /= grid->GetCellMass(incident); //Converting momentum to velocity
		printf("%f\n", grid->GetCellMass(incident));

		//Applying gravity to velocity
		Vector3D gravity_vector(0.0f, 0.0f, grid->gravity);
		grid->GetCellVelocity(incident) += (gravity_vector * grid->dt);

		//Boundary Conditions
		if (x_bounds < 2 || x_bounds > grid->side_size_ - 3) {
			grid->GetCellVelocity(incident).dim[0] = 0.0f;
		}
		if (y_bounds < 2 || y_bounds > grid->side_size_ - 3) {
			grid->GetCellVelocity(incident).dim[1] = 0.0f;
		}
		if (z_bounds < 2 || z_bounds > grid->side_size_ - 3) {
			grid->GetCellVelocity(incident).dim[2] = 0.0f;
		}
	}
}

__global__ void ClearGrid(Grid* grid) {
	//Cell Boundaries
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds); //Current position

	grid->GetCellMass(incident) = 0.0f;
	grid->GetCellVelocity(incident).Reset();
}

__global__ void SetValue(Grid* grid) {
	//Cell Boundaries
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds); //Current position

	Vector3D vector_pos((float) x_bounds / grid->GetResolution(), y_bounds, z_bounds);

	for (int i = 0; i < 3; i++) { //NOTE
		if (vector_pos.dim[i] > grid->side_size_ - 2) {
			vector_pos.dim[i] -= 2;
		}
		if (vector_pos.dim[i] < 2) {
			vector_pos.dim[i] += 2;
		}
	}

	grid->GetPosition(incident) = vector_pos;
	assert(vector_pos.dim[0] <= grid->side_size_ && vector_pos.dim[0] >= 0.0f);
	assert(vector_pos.dim[1] <= grid->side_size_ && vector_pos.dim[1] >= 0.0f);
	assert(vector_pos.dim[2] <= grid->side_size_ && vector_pos.dim[2] >= 0.0f);
}

__host__ void Grid::CalculateBounds() {
	unsigned int threads_per_dim = 8; //512 in total per block, max as each block can have 1024 threads only
	unsigned int block_count = ((side_size_ + threads_per_dim) - 1) / (threads_per_dim);

	cell_threads = dim3(threads_per_dim, threads_per_dim, threads_per_dim);
	cell_blocks = dim3(block_count, block_count, block_count);

	ProgramLog::OutputLine("Resolution: " + std::to_string(GetResolution()));

	//Resolution is now on blocks due to resolution being problematic with the thread count limits for blocks
	particle_threads = dim3(threads_per_dim, threads_per_dim, threads_per_dim);
	particle_blocks = dim3(block_count * GetResolution(), block_count, block_count);

	size_t thread_count = pow(block_count, 3) * pow(threads_per_dim, 3);

	ProgramLog::OutputLine("Threads (Total): " + std::to_string(thread_count));

	ProgramLog::OutputLine("Cells (Total): " + std::to_string(total_size_));
}

__host__ void GenerateRandomParticles(Grid* grid) {
	curandGenerator_t gen;
	srand(time(nullptr));

	int seed = rand();

	curandStatus_t curand_status;

	curand_status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	CurandCallCheck(curand_status);
	curand_status = curandSetPseudoRandomGeneratorSeed(gen, seed);
	CurandCallCheck(curand_status);
	curand_status = curandGenerateUniform(gen, grid->particle_mass_device_, grid->GetParticleCount());
	//curand_status = curandGenerateUniform(gen, (float*) grid->particle_position_device_, grid->GetParticleCount() * (sizeof(Vector3D) / sizeof(float)));
	CurandCallCheck(curand_status);
	curand_status = curandDestroyGenerator(gen);
	CurandCallCheck(curand_status);
	//Vector3D vec_test(1.0f, 1.0f, 1.0f);

	//cudaMemcpy(&grid->particle_position_device_[163], &vec_test, sizeof(Vector3D), cudaMemcpyHostToDevice);
}

__host__ static cudaError_t DebugGPU(Grid* grid, cudaStream_t& cuda_stream) {
	Vector3D* host_test = new Vector3D();
	cudaError_t cuda_status = cudaStreamSynchronize(cuda_stream); //WIP, DEBUG!

	cuda_status = cudaMemcpy(host_test, &grid->particle_position_device_[15563], sizeof(Vector3D), cudaMemcpyDeviceToHost); //WIP, DEBUG!

	s_stream << host_test->x() << " " << host_test->y() << " " << host_test->z() << std::endl; //WIP, DEBUG!
	ProgramLog::OutputLine(s_stream); //WIP, DEBUG!
	delete host_test;

	return cuda_status;
}

//Maybe make weights a shared memory allocation? Maybe a static shared memory context that is accessible through all the sim kernels?
__device__ Vector3D* GetWeights(Vector3D cell_difference) { //Returns weights shared
	Vector3D weights[3] {}; //Array of weights

	weights[0] = (cell_difference.Negative() + 0.5f).Squared() * 0.5f;
	weights[1] = cell_difference.Squared().Negative() + 0.75f;
	weights[2] = (cell_difference + 0.5f).Squared() * 0.5f;

	return weights;
}

__host__ cudaError_t Grid::SimulateGPU(Grid* grid, cudaStream_t& cuda_stream) {
	cudaError_t cuda_status = cudaSuccess;

	dim3& blocks = grid->cell_blocks, &threads = grid->cell_threads;
	dim3& blocks2 = grid->particle_blocks, &threads2 = grid->particle_threads;

	if (!grid->up_to_date_) {
		grid->CalculateBounds();
		grid->DeviceTransfer(grid);

		GenerateRandomParticles(grid);

		SetValue<<<blocks2, threads2, 0, cuda_stream>>> (grid);
		cuda_status = cudaStreamSynchronize(cuda_stream);

		grid->up_to_date_ = true;
	}

	ProgramLog::OutputLine("\n\nBREAK\n\n");

	ClearGrid<<<blocks, threads, 0, cuda_stream>>> (grid);

	cuda_status = PostExecutionChecks(cuda_status, "ClearGrid", false);
	CudaExceptionHandler(cuda_status, "ClearGrid failed!");

	DebugGPU(grid, cuda_stream);

	UpdateCell<<<blocks2, threads2, 0, cuda_stream>>> (grid);
	cuda_status = PostExecutionChecks(cuda_status, "CellMomentum", false);
	CudaExceptionHandler(cuda_status, "CellMomentum failed!");

	DebugGPU(grid, cuda_stream);

	SimulateGrid<<<blocks2, threads2, 0, cuda_stream>>> (grid); //NOTE
	cuda_status = PostExecutionChecks(cuda_status, "VelocityGradientSolve", false);
	CudaExceptionHandler(cuda_status, "SimulateGrid failed!");

	DebugGPU(grid, cuda_stream);

	UpdateGrid<<<blocks, threads, 0, cuda_stream>>> (grid);
	cuda_status = PostExecutionChecks(cuda_status, "UpdateGrid", false);
	CudaExceptionHandler(cuda_status, "UpdateGrid failed!");

	DebugGPU(grid, cuda_stream);

	AdvectParticles<<<blocks2, threads2, 0, cuda_stream>>> (grid);
	cuda_status = PostExecutionChecks(cuda_status, "AdvectParticles", false);
	CudaExceptionHandler(cuda_status, "AdvectParticles failed!");

	cudaStreamSynchronize(cuda_stream);

	DebugGPU(grid, cuda_stream);

	return cuda_status;
}