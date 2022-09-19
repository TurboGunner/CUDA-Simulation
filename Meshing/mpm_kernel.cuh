#pragma once

#include "matrix.cuh"
#include "mpm.cuh"
#include "vector_cross.cuh"

#include "../CUDATest/index_pair_cuda.cuh"

#include "../CUDATest/handler_methods.hpp"

#include <iostream>

__global__ void InitializeGrid(Grid* grid, float resolution) {
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;
	
	IndexPair incident(x_bounds, y_bounds, z_bounds);

	Cell* cell = new Cell();
	grid->AddCell(cell, incident.IX(grid->side_size_));

	for (int i = 0; i < resolution; i++) {
		Particle* particle = new Particle();
		grid->AddParticle(particle, incident.IX(grid->side_size_) * i);
	}
}

__host__ cudaError_t Grid::SimulateGPU(Grid* grid) {
	cudaError_t cuda_status = cudaSuccess;

	cuda_status = grid->DeviceTransfer(grid);
	
	dim3 blocks, threads;

	ThreadAllocator(blocks, threads, grid->side_size_);
	InitializeGrid<<<blocks, threads>>> (grid->device_alloc_, grid->resolution_);

	cuda_status = PostExecutionChecks(cuda_status, "SimulateGPU", true);

	return cuda_status;
}