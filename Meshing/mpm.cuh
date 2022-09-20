#pragma once

#include "matrix.cuh"
#include "vector_cross.cuh"

#include "../CUDATest/index_pair_cuda.cuh"

#include "../CUDATest/handler_methods.hpp"

#include <stdio.h>
#include <stdlib.h>

struct Particle {
	Vector3D position;
	Vector3D velocity;
	float mass = 0.0f;
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

	__host__ __device__ size_t GetParticleCount() const;

	__host__ __device__ float GetResolution() const;

	__host__ __device__ void AddCell(Cell* cell, const size_t& index);

	__host__ __device__ void AddParticle(Particle* particle, const size_t& index);

	__host__ __device__ Cell* GetCell(const size_t& index);

	__host__ __device__ Particle* GetParticle(const size_t& index);

	__host__ __device__ Particle* GetParticle(IndexPair incident);

	const float gravity = -0.3f;

	const float rest_density = 4.0f;
	const float dynamic_viscosity = 0.1f;

	const float eos_stiffness = 10.0f;
	const float eos_power = 4;

	const float dt = 0.2f;

	Grid* device_alloc_;

	size_t side_size_;

private:
	Particle** particles_, **particles_device_;
	Cell** cells_, **cells_device_;

	Vector3D sim_size_;
	size_t total_size_;

	bool device_allocated_status = false, is_initialized_ = false;

	float resolution_;
};

__global__ void InitializeGrid(Grid* grid);

__global__ void SimulateGrid(Grid* grid);

__global__ void UpdateCell(Grid* grid);