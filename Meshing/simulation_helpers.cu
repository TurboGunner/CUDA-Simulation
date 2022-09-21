#include "mpm.cuh"

__global__ void InitializeGrid(Grid* grid) {
	//Cell Boundaries
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds);

	Cell* cell = new Cell();
	cell->velocity = Vector3D(0.0f, 0.0f, 0.0f);
	cell->mass = 0.0f;
	grid->AddCell(cell, incident.IX(grid->side_size_));

	for (size_t i = 0; i < grid->GetResolution(); i++) { //IDX, Particle
		Particle* particle = new Particle();
		particle->position = Vector3D(0.0f, 0.0f, 0.0f);
		particle->velocity = Vector3D(0.0f, 0.0f, 0.0f);
		particle->momentum = Matrix(3, 3, false);
		particle->mass = 0.0f;

		grid->AddParticle(particle, incident.IX(grid->side_size_) + i);
	}
}

__host__ cudaError_t InitializeGridHost(Grid* grid) {
	for (size_t i = 0; i < grid->GetTotalSize(); i++) {
		Cell* cell = new Cell();
		cell->velocity = Vector3D(0.0f, 0.0f, 0.0f);
		cell->mass = 0.0f;
		grid->AddCell(cell, i);
	}


	for (size_t j = 0; j < grid->GetParticleCount(); j++) { //IDX, Particle
		Particle* particle = new Particle();
		particle->position = Vector3D(0.0f, 0.0f, 0.0f);
		particle->velocity = Vector3D(0.0f, 0.0f, 0.0f);
		particle->momentum = Matrix(3, 3, false);
		grid->AddParticle(particle, j);
	}
	return grid->DeviceTransfer(grid);
}

__global__ void UpdateGrid(Grid* grid) {
	//Cell Boundaries
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds); //Current Index

	Cell* cell = grid->GetCell(incident); //Current Cell

	cell->velocity = cell->velocity / cell->mass; //Converting momentum to velocity

	//Applying gravity to velocity
	Vector3D gravity_vector(0.0f, 0.0f, grid->gravity);
	cell->velocity = cell->velocity + (gravity_vector * grid->dt);

	//Boundary Conditions
	if (x_bounds < 2 || x_bounds > grid->side_size_ - 3) {
		cell->velocity.dim[0] = 0;
	}
	if (y_bounds < 2 || y_bounds > grid->side_size_ - 3) {
		cell->velocity.dim[1] = 0;
	}
	if (z_bounds < 2 || z_bounds > grid->side_size_ - 3) {
		cell->velocity.dim[2] = 0;
	}
}

__device__ Vector3D* GetWeights(Vector3D cell_difference) { //Returns weights shared
	Vector3D weights[3]{}; //Array of weights

	weights[0] = (cell_difference.Negative() + 0.5f).Squared() * 0.5f;
	weights[1] = cell_difference.Squared().Negative() + 0.75f;
	weights[2] = (cell_difference + 0.5f).Squared() * 0.5f;

	return weights;
}

__device__ IndexPair* GetTraversals(IndexPair incident) {
	IndexPair incidents[27] = { incident, //D3Q27 tensor traversal
		incident.Left(), incident.Right(), incident.Front(), incident.Back(), incident.Up(), incident.Down(),
		incident.CornerLDownFront(), incident.CornerLDownBack(), incident.CornerRDownFront(), incident.CornerRDownBack(),
		incident.CornerLUpFront(), incident.CornerLUpBack(), incident.CornerRUpFront(), incident.CornerRUpBack(),
		incident.CornerLMidBack(), incident.CornerRMidBack(), incident.CornerLMidFront(), incident.CornerRMidFront(),
		incident.MidUpFront(), incident.MidUpBack(), incident.MidUpLeft(), incident.MidUpRight(),
		incident.MidDownFront() ,incident.MidDownBack(), incident.MidDownLeft(), incident.MidDownRight()
	};
	return incidents;
}

__global__ void ClearGrid(Grid* grid) {
	//Cell Boundaries
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds); //Current position

	Cell* cell = grid->GetCell(incident);

	cell->mass = 0;
	cell->velocity.Reset();
}

__host__ cudaError_t Grid::SimulateGPU(Grid* grid) {
	cudaError_t cuda_status = cudaSuccess;

	cuda_status = InitializeGridHost(grid);

	std::cout << "Allocated successfully " << grid->GetTotalSize() << " cells! (host)" << std::endl;
	std::cout << "Allocated successfully " << grid->GetParticleCount() << " particles! (host)" << std::endl;

	dim3 blocks, threads;
	unsigned int threads_per_dim = (unsigned int)cbrt (64);
	unsigned int block_count = ((grid->side_size_ + threads_per_dim) - 1) / (threads_per_dim);

	threads = dim3(threads_per_dim, threads_per_dim, threads_per_dim);
	blocks = dim3(block_count, block_count, block_count);

	dim3 blocks2, threads2;

	threads2 = dim3(threads_per_dim * grid->GetResolution(), threads_per_dim * grid->GetResolution(), threads_per_dim * grid->GetResolution());
	blocks2 = dim3(block_count * grid->GetResolution(), block_count * grid->GetResolution(), block_count * grid->GetResolution());

	InitializeGrid<<<blocks, threads>>> (grid->device_alloc_);
	cuda_status = PostExecutionChecks(cuda_status, "GridInitialization", true);

	std::cout << "Allocated successfully " << grid->GetTotalSize() << " cells! (device)" << std::endl;
	std::cout << "Allocated successfully " << grid->GetParticleCount() << " particles! (device)" << std::endl;

	UpdateCell<<<blocks, threads>>> (grid->device_alloc_);
	cuda_status = PostExecutionChecks(cuda_status, "CellMomentum", true);

	std::cout << "Ran through cell momentum!" << std::endl;

	SimulateGrid<<<blocks2, threads2>>> (grid->device_alloc_);
	cuda_status = PostExecutionChecks(cuda_status, "VelocityGradientSolve", true);

	std::cout << "Ran through the velocity gradient solve!" << std::endl;

	//UpdateGrid<<<blocks, threads>>> (grid->device_alloc_);
	cuda_status = PostExecutionChecks(cuda_status, "UpdateGrid", true);

	std::cout << "Updated the grid!" << std::endl;

	//AdvectParticles<<<blocks2, threads2>>> (grid->device_alloc_);
	cuda_status = PostExecutionChecks(cuda_status, "AdvectParticles", true);

	std::cout << "Advected the particles!" << std::endl;

	cuda_status = grid->HostTransfer();

	return cuda_status;
}