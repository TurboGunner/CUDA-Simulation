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

	for (size_t i = 0; i < grid->GetParticleCount(); i++) { //IDX, Particle
		Particle* particle = new Particle();
		particle->position = Vector3D(0.0f, 0.0f, 0.0f);
		particle->velocity = Vector3D(0.0f, 0.0f, 0.0f);
		particle->momentum = Matrix(3, 3, false);
		particle->mass = 0.0f;

		grid->AddParticle(particle, i);
	}

	return grid->DeviceTransfer(grid);
}