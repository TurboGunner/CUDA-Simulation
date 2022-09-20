#pragma once

#include "matrix.cuh"
#include "mpm.cuh"
#include "vector_cross.cuh"

#include "../CUDATest/index_pair_cuda.cuh"

#include "../CUDATest/handler_methods.hpp"

#include <iostream>
#include <math.h>

__global__ void InitializeGrid(Grid* grid) {
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;
	
	IndexPair incident(x_bounds, y_bounds, z_bounds);

	Cell* cell = new Cell();
	grid->AddCell(cell, incident.IX(grid->side_size_));

	for (int i = 0; i < grid->GetResolution(); i++) {
		Particle* particle = new Particle();
		grid->AddParticle(particle, incident.IX(grid->side_size_) * i);
	}
}

__global__ void SimulateGrid(Grid* grid) {
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds);

	Particle* particle = grid->GetParticle(incident);
	Vector3D position = particle->position;
	Vector3D cell_idx = position.Truncate();

	Vector3D cell_difference = (position - cell_idx) - 0.5f;

	Vector3D weights[3] {};

	weights[0] = (cell_difference.Negative() + 0.5f).Squared() * 0.5f;
	weights[1] = cell_difference.Squared().Negative() + 0.75f;
	weights[2] = (cell_difference + 0.5f).Squared() * 0.5f;

	const size_t traversal_length = 19;

	IndexPair incidents[traversal_length] = { incident,
		incident.Left(), incident.Right(), incident.Front(), incident.Back(), incident.Up(), incident.Down(),
		incident.CornerLDownFront(), incident.CornerLDownBack(), incident.CornerRDownFront(), incident.CornerRDownBack(),
		incident.CornerLUpFront(), incident.CornerLUpBack(), incident.CornerRUpFront(), incident.CornerRUpBack(),
		incident.CornerLMidBack(), incident.CornerRMidBack(), incident.CornerLMidFront(), incident.CornerRMidFront()
	};

	float density = 0.0f;
	for (size_t i = 0; i < traversal_length; ++i) {
		int x_weight_idx = (i / 6) % 2,
			y_weight_idx = (i / 3) % 2,
			z_weight_idx = i % 2;

		float weight = weights[x_weight_idx].x() * weights[y_weight_idx].y() * weights[z_weight_idx].z();
		density += grid->GetParticle(incidents[i])->mass * weight;
	}

	float volume = particle->mass / density;

	float pressure = fmax(-0.1f, grid->eos_stiffness * (pow(density / grid->rest_density, grid->eos_power) - 1));

	float points[3] = { -pressure, -pressure, -pressure };

	Matrix* stress_matrix = Matrix::DiagonalMatrix(points, 3, 3);

	Matrix& strain_matrix = particle->momentum;

	float trace = strain_matrix.Get(2, 0) + strain_matrix.Get(1, 1) + strain_matrix.Get(0, 2);
	for (size_t i = 0; i < strain_matrix.rows; i++) {
		size_t reverse_idx = strain_matrix.rows - i - 1;
		strain_matrix.Get(i, reverse_idx) = trace;
	}

	Matrix viscosity_term = strain_matrix * grid->dynamic_viscosity;

	Matrix::AddOnPointer(stress_matrix, viscosity_term);

	Matrix* eq_16_term_0 = Matrix::Create(stress_matrix->rows, stress_matrix->columns);
	Matrix::MultiplyScalarOnPointer(eq_16_term_0, -volume * 4 * grid->dt);

	for (size_t i = 0; i < traversal_length; ++i) {
		int x_weight_idx = (i / 6) % 2,
			y_weight_idx = (i / 3) % 2,
			z_weight_idx = i % 2;

		float weight = weights[x_weight_idx].x() * weights[y_weight_idx].y() * weights[z_weight_idx].z();
		Vector3D cell_position(cell_idx.x() + x_weight_idx - 1,
			cell_idx.y() + y_weight_idx - 1,
			cell_idx.z() + z_weight_idx - 1);

		Vector3D cell_dist = (cell_position - position) + 0.5f;
		int cell_index = incident.IX(grid->side_size_ / grid->GetResolution());
		Cell* cell = grid->GetCell(cell_index);

		Matrix::MultiplyScalarOnPointer(eq_16_term_0, weight);

		Matrix* cell_dist_matrix = Matrix::Create(1, 3);
		*cell_dist_matrix = cell_dist.ToMatrix();

		Matrix* momentum = Matrix::Create(1, 3);

		MultiplyKernel(eq_16_term_0, cell_dist_matrix, momentum);

		Vector3D momentum_vector(momentum->Get(0), momentum->Get(1), momentum->Get(2));

		cell->velocity = cell->velocity + momentum_vector;
	}
}

__host__ cudaError_t Grid::SimulateGPU(Grid* grid) {
	cudaError_t cuda_status = cudaSuccess;

	cuda_status = grid->DeviceTransfer(grid);
	
	dim3 blocks, threads;
	ThreadAllocator(blocks, threads, grid->side_size_);

	InitializeGrid<<<blocks, threads>>> (grid->device_alloc_);

	cuda_status = PostExecutionChecks(cuda_status, "SimulateGPU", true);

	std::cout << "Allocated successfully " << grid->GetTotalSize() << " cells!" << std::endl;
	std::cout << "Allocated successfully " << grid->GetParticleCount() << " particles!" << std::endl;

	return cuda_status;
}