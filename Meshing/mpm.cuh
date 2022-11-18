#pragma once

#include "matrix.cuh"
#include "vector_cross.cuh"

#include "curand.h"
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

	__host__ Grid(const Vector3D& sim_size_in, const float resolution_in = 4.0f, const bool late_init_in = true) {
		sim_size_ = sim_size_in;

		sim_size_ = Vector3D(truncf(sim_size_.x()), truncf(sim_size_.y()), truncf(sim_size_.z()));

		for (int i = 0; i < 3; i++) {
			assert(sim_size_.dim[i] >= 1.0f);
		}

		side_size_ = sim_size_.x();

		total_size_ = sim_size_.x() * sim_size_.y() * sim_size_.z();

		if (resolution_in <= 0.0f) {
			std::cout << "\n\nWarning! The resolution parameter should be a positive number greater than zero. Set to default of 4 to prevent segfault." << std::endl;
			resolution_ = 4.0f;
		}
		else {
			resolution_ = resolution_in;
		}

		late_init_ = late_init_in;

		cudaError_t cuda_status = cudaSuccess;

		if (!late_init_) {
			cuda_status = cudaMalloc(&particle_position_device_, GetParticleCount() * sizeof(Vector3D));
		}
		if (host_sync_) {
			cuda_status = cudaMallocHost(&particle_position_, GetParticleCount() * sizeof(Vector3D));
		}

		cuda_status = cudaMalloc(&particle_velocity_device_, GetParticleCount() * sizeof(Vector3D));
		cuda_status = cudaMalloc(&particle_mass_device_, GetParticleCount() * sizeof(float));

		cuda_status = cudaMalloc(&momentum_matrices_, GetParticleCount() * sizeof(Matrix<3, 3>));

		cuda_status = cudaMalloc(&cell_velocity_device_, total_size_ * sizeof(Vector3D)); //NOTE
		cuda_status = cudaMalloc(&cell_mass_device_, total_size_ * sizeof(float));

		is_initialized_ = true;
	}

	__host__ ~Grid() {
		cudaError_t cuda_status = cudaFree(particle_position_device_);
		cuda_status = cudaFree(particle_velocity_device_);
		cuda_status = cudaFree(particle_mass_device_);

		cuda_status = cudaFree(cell_velocity_device_);
		cuda_status = cudaFree(cell_mass_device_);

		if (host_sync_) {
			cuda_status = cudaFreeHost(particle_position_);
		}
	}

	__host__ void* operator new(size_t size) {
		void* ptr;
		cudaMallocHost(&ptr, sizeof(Grid)); //Declares page-locked memory for the initial allocation
		return ptr;
	}

	__host__ void operator delete(void* ptr) {
		cudaFreeHost(ptr);
	}

	__host__ cudaError_t DeviceTransfer(Grid*& src) {
		cudaError_t cuda_status = cudaSuccess;

		if (!device_allocated_status) {
			cuda_status = cudaMalloc(&device_alloc_, sizeof(Grid));
			device_allocated_status = true;
			cuda_status = CopyFunction("DeviceTransferObject", device_alloc_, src, cudaMemcpyHostToDevice, cuda_status, sizeof(Grid), 1);
		}
		return cuda_status;
	}

	__host__ cudaError_t HostTransfer() { //NOTE
		cudaError_t cuda_status = cudaSuccess;
		cuda_status = CopyFunction("HostTransferParticlesPosition", particle_position_, particle_position_device_, cudaMemcpyDeviceToHost, cuda_status, sizeof(Vector3D), GetParticleCount());
		cuda_status = cudaDeviceSynchronize();

		return cuda_status;
	}

	__host__ static cudaError_t SimulateGPU(Grid* grid, cudaStream_t& cuda_stream);

	__host__ __device__ size_t GetTotalSize() const {
#ifdef __CUDA_ARCH__
		return device_alloc_->total_size_;
#else
		return total_size_;
#endif
	}

	__host__ __device__ size_t GetParticleCount() const {
#ifdef __CUDA_ARCH__
		return device_alloc_->total_size_ * device_alloc_->resolution_;
#else
		return total_size_ * resolution_;
#endif
	}

	__host__ __device__ float GetResolution() const {
#ifdef __CUDA_ARCH__
		return device_alloc_->resolution_;
#else
		return resolution_;
#endif
	}

	__host__ __device__ Vector3D& GetVelocity(const size_t index) {
		assert(index <= GetParticleCount());
#ifdef __CUDA_ARCH__
		return particle_velocity_device_[index];
#else
		return particle_velocity_[index];
#endif
	}

	__host__ __device__ Vector3D& GetVelocity(IndexPair incident) {
		uint3 index_size{ side_size_ * resolution_, side_size_, side_size_ };
		size_t index = incident.IX(index_size);
		return GetVelocity(index);
	}

	__host__ __device__ Vector3D& GetPosition(const size_t index) {
		assert(index <= GetParticleCount());
#ifdef __CUDA_ARCH__
		return particle_position_device_[index];
#else
		return particle_position_[index];
#endif
	}

	__host__ __device__ Vector3D& GetPosition(IndexPair incident) {
		uint3 index_size{ side_size_ * resolution_, side_size_, side_size_ };
		size_t index = incident.IX(index_size);
		return GetPosition(index);
	}


	__host__ __device__ Matrix<3, 3>& GetMomentum(const size_t index) { //NOTE: DEVICE ONLY!
		assert(index <= GetParticleCount());
		return momentum_matrices_[index];
	}

	__host__ __device__ Matrix<3, 3>& GetMomentum(IndexPair incident) {
		uint3 index_size{ side_size_ * resolution_, side_size_, side_size_ };
		size_t index = incident.IX(index_size);
		return GetMomentum(index);
	}

	__host__ __device__ float& GetParticleMass(const size_t index) {
		assert(index <= GetParticleCount());
#ifdef __CUDA_ARCH__
		return particle_mass_device_[index];
#else
		return particle_mass_[index];
#endif
	}

	__host__ __device__ float& GetParticleMass(IndexPair incident) {
		uint3 index_size{ side_size_ * resolution_, side_size_, side_size_ };
		size_t index = incident.IX(index_size);
		return GetParticleMass(index);
	}

	__host__ __device__ float& GetCellMass(const size_t index) {
		assert(index <= total_size_);
#ifdef __CUDA_ARCH__
		return cell_mass_device_[index];
#else
		return cell_mass_[index];
#endif
	}

	__host__ __device__ float& GetCellMass(IndexPair incident) {
		size_t index = incident.IX(side_size_);
		return GetCellMass(index);
	}

	__host__ __device__ Vector3D& GetCellVelocity(const size_t index) {
		assert(index <= total_size_);
#ifdef __CUDA_ARCH__
		return cell_velocity_device_[index];
#else
		return cell_velocity_[index];
#endif
	}

	__host__ __device__ Vector3D& GetCellVelocity(IndexPair incident) {
		assert(incident.x <= side_size_ - 1);
		assert(incident.y <= side_size_ - 1);
		assert(incident.z <= side_size_ - 1);

		size_t index = incident.IX(side_size_);
		return GetCellVelocity(index);
	}

	__host__ void CalculateBounds();

	float gravity = -0.3f;

	float rest_density = 4.0f;
	float dynamic_viscosity = 0.1f;

	float eos_stiffness = 10.0f;
	float eos_power = 4.0f;

	float dt = 0.2f;

	Grid* device_alloc_;

	size_t side_size_;

	Vector3D* particle_position_, *particle_position_device_;
	Vector3D* particle_velocity_, *particle_velocity_device_;
	float* particle_mass_, *particle_mass_device_;

	Vector3D* cell_velocity_, *cell_velocity_device_;
	float* cell_mass_, *cell_mass_device_;

	Matrix<3, 3>* momentum_matrices_;

	bool host_sync_ = false;
	bool up_to_date_ = false;
	bool late_init_ = true;

private:
	Vector3D sim_size_;
	size_t total_size_ = 0;

	bool device_allocated_status = false, is_initialized_ = false;

	float resolution_;

	dim3 cell_blocks, cell_threads;
	dim3 particle_blocks, particle_threads;
};

//Globals

__global__ void UpdateCell(Grid* grid);

__global__ void SimulateGrid(Grid* grid); //Stress matrix must be diagonal!

__global__ void UpdateGrid(Grid* grid);

__global__ void AdvectParticles(Grid* grid);
__device__ void MPMBoundaryConditions(Grid* grid, IndexPair incident, const Vector3D& position_normalized, const float wall_min, const float wall_max);

__global__ void ClearGrid(Grid* grid);

__global__ void SetValue(Grid* grid);

//Device Methods

__device__ Vector3D* GetWeights(Vector3D cell_difference);

__host__ static void GenerateRandomParticles(Grid* grid);

__forceinline__ __device__ Vector3D UnrolledFixedMV(const Matrix<3, 3>& matrix, const Vector3D& vector) {
	Vector3D output = {};

	output.dim[0] = matrix.Get(0) * vector.dim[0];
	output.dim[0] += matrix.Get(1) * vector.dim[1];
	output.dim[0] += matrix.Get(2) * vector.dim[2];

	output.dim[1] = matrix.Get(3) * vector.dim[0];
	output.dim[1] += matrix.Get(4) * vector.dim[1];
	output.dim[1] += matrix.Get(5) * vector.dim[2];

	output.dim[2] = matrix.Get(6) * vector.dim[0];
	output.dim[2] += matrix.Get(7) * vector.dim[1];
	output.dim[2] += matrix.Get(8) * vector.dim[2];

	return output;
}