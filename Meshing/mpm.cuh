#pragma once

#include "matrix.cuh"
#include "vector_cross.cuh"

#include "curand_kernel.h"

#include "../CUDATest/index_pair_cuda.cuh"

#include "../CUDATest/handler_methods.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>

class Grid {
public:
	Grid() = default;

	__host__ Grid(const Vector3D& sim_size_in, const float& resolution_in = 4.0f);

	__host__ ~Grid();

	__host__ void* operator new(size_t size);

	__host__ void operator delete(void* ptr);

	__host__ cudaError_t DeviceTransfer(Grid*& src);

	__host__ cudaError_t HostTransfer();

	__host__ static cudaError_t SimulateGPU(Grid* grid, cudaStream_t& cuda_stream);

	__host__ __device__ size_t GetTotalSize() const;

	__host__ __device__ size_t GetParticleCount() const;

	__host__ __device__ float GetResolution() const;

	__host__ __device__ Vector3D& GetVelocity(const size_t& index);
								 
	__host__ __device__ Vector3D& GetVelocity(IndexPair incident);

	__host__ __device__ Vector3D& GetPosition(const size_t& index);

	__host__ __device__ Vector3D& GetPosition(IndexPair incident);

	__host__ __device__ Matrix& GetMomentum(const size_t& index);

	__host__ __device__ Matrix& GetMomentum(IndexPair incident);

	__host__ __device__ float& GetParticleMass(const size_t& index);

	__host__ __device__ float& GetParticleMass(IndexPair incident);

	__host__ __device__ float& GetCellMass(const size_t& index);

	__host__ __device__ float& GetCellMass(IndexPair incident);

	__host__ __device__ Vector3D& GetCellVelocity(const size_t& index);

	__host__ __device__ Vector3D& GetCellVelocity(IndexPair incident);

	__host__ void CalculateBounds();

	float gravity = -0.3f;

	float rest_density = 4.0f;
	float dynamic_viscosity = 0.1f;

	float eos_stiffness = 10.0f;
	float eos_power = 4;

	float dt = 0.2f;

	Grid* device_alloc_;

	size_t side_size_;

	Vector3D* particle_position_, *particle_position_device_;
	Vector3D* particle_velocity_, *particle_velocity_device_;
	float* particle_mass_, *particle_mass_device_;

	Vector3D* cell_velocity_, *cell_velocity_device_;
	float* cell_mass_, *cell_mass_device_;

	Matrix* momentum_matrices_;

	bool host_sync_ = false;
	bool up_to_date_ = false;

private:
	Vector3D sim_size_;
	size_t total_size_;

	bool device_allocated_status = false, is_initialized_ = false;

	float resolution_;

	Matrix* cell_dist_matrix = Matrix::Create(3, 1, false);
	Matrix* momentum = Matrix::Create(3, 1, false);
	Matrix* momentum_matrix = Matrix::Create(3, 3, false);

	Matrix* stress_matrix = Matrix::Create(3, 3, false);
	Matrix* weighted_stress = Matrix::Create(3, 3, false);

	Matrix* B_term = Matrix::Create(3, 3, false);
	Matrix* weighted_term = Matrix::Create(3, 3, false);

	Matrix* viscosity_term = Matrix::Create(3, 3, false);

	dim3 cell_blocks, cell_threads;
	dim3 particle_blocks, particle_threads;
};

//Globals

__global__ void UpdateCell(Grid* grid, Matrix* momentum_matrix, Matrix* cell_dist_matrix, Matrix* momentum);

__global__ void SimulateGrid(Grid* grid, Matrix* stress_matrix, Matrix* weighted_stress, Matrix* cell_dist_matrix, Matrix* momentum, Matrix* viscosity_term); //Stress matrix must be diagonal!

__global__ void UpdateGrid(Grid* grid);

__global__ void AdvectParticles(Grid* grid, Matrix* B_term, Matrix* weighted_term);
__device__ void MPMBoundaryConditions(Grid* grid, IndexPair incident, const Vector3D& position_normalized, const float& wall_min, const float& wall_max);

__global__ void ClearGrid(Grid* grid);

__host__ cudaError_t InitializeGridHost(Grid* grid);

//Device Methods

__device__ Vector3D* GetWeights(Vector3D cell_difference);