#include "mpm.cuh"

#include <iostream>

__host__ Grid::Grid(const Vector3D& sim_size_in, const float& resolution_in) { //Make sure these are integers!
	sim_size_ = sim_size_in;

	side_size_ = sim_size_.x();

	total_size_ = sim_size_.x() * sim_size_.y() * sim_size_.z();

	if (resolution_in <= 0.0f) {
		std::cout << "\n\nWarning! The resolution parameter should be a positive greater than 0.0. Set to default of 4 to prevent segfault." << std::endl;
		resolution_ = 4.0f;
	}
	else {
		resolution_ = resolution_in;
	}

	cudaError_t cuda_status = cudaSuccess;

	cudaMalloc(&particles_device_, sizeof(Particle*) * total_size_ * resolution_);
	cudaMallocHost(&particles_, sizeof(Particle*) * total_size_ * resolution_);

	cudaMalloc(&cells_device_, sizeof(Cell*) * total_size_);
	cudaMallocHost(&cells_, sizeof(Cell*) * total_size_);

	is_initialized_ = true;
}

__host__ void* Grid::operator new(size_t size) {
	void* ptr;
	cudaMallocHost(&ptr, sizeof(Grid));
	return ptr;
}

__host__ void Grid::operator delete(void* ptr) {
	free(ptr);
}

__host__ cudaError_t Grid::DeviceTransfer(Grid*& src) {
	cudaError_t cuda_status = cudaSuccess;

	cuda_status = CopyFunction("DeviceTransferParticles", particles_device_, particles_, cudaMemcpyHostToDevice, cuda_status, sizeof(Particle*), total_size_);
	cuda_status = CopyFunction("DeviceTransferCells", cells_device_, cells_, cudaMemcpyHostToDevice, cuda_status, sizeof(Cell*), total_size_);

	if (!device_allocated_status) {
		cuda_status = cudaMalloc(&device_alloc_, sizeof(Grid));
		device_allocated_status = true;
		cuda_status = CopyFunction("DeviceTransferObject", device_alloc_, src, cudaMemcpyHostToDevice, cuda_status, sizeof(Grid), 1);
	}
	return cuda_status;
}

__host__ void Grid::HostTransfer(cudaError_t& cuda_status) {
	cuda_status = CopyFunction("DeviceTransferParticles", particles_, particles_device_, cudaMemcpyDeviceToHost, cuda_status, sizeof(Particle*), total_size_);
	cuda_status = CopyFunction("DeviceTransferCells", cells_, cells_device_, cudaMemcpyDeviceToHost, cuda_status, sizeof(Cell*), total_size_);
	cudaDeviceSynchronize();
}

__host__ __device__ size_t Grid::GetTotalSize() const {
	return total_size_;
}

__host__ __device__ size_t Grid::GetParticleCount() const {
	return total_size_ * resolution_;
}

__host__ __device__ float Grid::GetResolution() const {
	return resolution_;
}

__host__ __device__ void Grid::AddCell(Cell* cell, const size_t& index) {
#ifdef __CUDA_ARCH__
	cells_device_[index] = cell;
#else
	cells_[index] = cell;
#endif
}

__host__ __device__ void Grid::AddParticle(Particle* particle, const size_t& index) {
#ifdef __CUDA_ARCH__
	particles_device_[index] = particle;
#else
	particles_[index] = particle;
#endif
}

__host__ __device__ Cell* Grid::GetCell(const size_t& index) {
	if (index >= total_size_) {
		printf("%s %zu\n", "Warning! Out of bounds access. Input Index: ", index);
	}
#ifdef __CUDA_ARCH__
	return cells_device_[index];
#else
	return cells_[index];
#endif
}

__host__ __device__ Cell* Grid::GetCell(IndexPair incident) {
	size_t index = incident.IX(side_size_);
	if (index >= total_size_) {
		printf("%s %zu\n", "Warning! Out of bounds access. Input Index: ", index);
	}
#ifdef __CUDA_ARCH__
	return cells_device_[index];
#else
	return cells_[index];
#endif
}

__host__ __device__ Particle* Grid::GetParticle(const size_t& index) {
	if (index >= GetParticleCount()) {
		printf("%s %zu\n", "Warning! Out of bounds access. Input Index: ", index);
	}
#ifdef __CUDA_ARCH__
	return particles_device_[index];
#else
	return particles_[index];
#endif
}

__host__ __device__ Particle* Grid::GetParticle(IndexPair incident) {
	size_t index = incident.IX(side_size_ * resolution_); //NOTE
	if (index >= GetParticleCount()) {
		printf("%s %zu\n", "Warning! Out of bounds access. Input Index: ", index);
	}
#ifdef __CUDA_ARCH__
	return particles_device_[index];
#else
	return particles_[index];
#endif
}