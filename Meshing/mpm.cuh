#pragma once

#include "matrix.cuh"
#include "vector_cross.cuh"

#include "../CUDATest/handler_methods.hpp"

#include <stdio.h>
#include <stdlib.h>

struct Particle {
	Vector3D position;
	Vector3D velocity;
	Matrix momentum = Matrix(3, 3, true);
};

struct Cell {
	Vector3D velocity;
	float mass = 0.0f;
};

class Grid {
public:
	Grid() = default;

	__host__ Grid(const Vector3D& sim_size_in, const float& resolution_in = 4.0f);

	__host__ void* operator new(size_t size);

	__host__ void operator delete(void* ptr);

	__host__ cudaError_t DeviceTransfer(Grid*& src);

	__host__ void HostTransfer(cudaError_t& cuda_status);

	__host__ static cudaError_t SimulateGPU(Grid* grid);

	__host__ __device__ size_t GetTotalSize() const;

	Grid* device_alloc_;

	size_t side_size_;

	__host__ __device__ void AddCell(Cell* cell, const size_t& index);
	__host__ __device__ void AddParticle(Particle* particle, const size_t& index);

private:
	Particle** particles_, **particles_device_;
	Cell** cells_, **cells_device_;

	Vector3D sim_size_;
	size_t total_size_;

	bool device_allocated_status = false, is_initialized_ = false;

	float resolution_;
};

__global__ void InitializeGrid(Grid* grid, float resolution);