#include "mpm.cuh"

#include <iostream>

__host__ Grid::Grid(const Vector3D& sim_size_in, const float resolution_in, const bool late_init_in) { //Make sure these are integers!
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

	late_init_ = late_init_in;

	cudaError_t cuda_status = cudaSuccess;
	
	if (!late_init_) {
		cuda_status = cudaMalloc(&particle_position_device_, GetParticleCount() * sizeof(Vector3D));
		if (host_sync_) {
			cuda_status = cudaMallocHost(&particle_position_, GetParticleCount() * sizeof(Vector3D));
		}
	}

	cuda_status = cudaMalloc(&particle_velocity_device_, GetParticleCount() * sizeof(Vector3D));
	cuda_status = cudaMalloc(&particle_mass_device_, GetParticleCount() * sizeof(float));

	cuda_status = cudaMalloc(&cell_velocity_device_, total_size_ * sizeof(Vector3D)); //NOTE
	//cudaMallocHost(&cells_, sizeof(Cell));
	cuda_status = cudaMalloc(&cell_mass_device_, total_size_ * sizeof(float));

	momentum_matrices_ = Matrix::MatrixMassAllocation(GetParticleCount(), 3, 3);

	is_initialized_ = true;
}

__host__ Grid::~Grid() {
	cudaError_t cuda_status = cudaFree(particle_position_device_);
	cuda_status = cudaFree(particle_velocity_device_);
	cuda_status = cudaFree(particle_mass_device_);

	cuda_status = cudaFree(cell_velocity_device_);
	cuda_status = cudaFree(cell_mass_device_);

	if (host_sync_) {
		cuda_status = cudaFreeHost(particle_position_);
	}
	for (size_t i = 0; i < GetParticleCount(); i++) {
		momentum_matrices_[0].Destroy();
	}
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

	if (!device_allocated_status) {
		cuda_status = cudaMalloc(&device_alloc_, sizeof(Grid));
		device_allocated_status = true;
		cuda_status = CopyFunction("DeviceTransferObject", device_alloc_, src, cudaMemcpyHostToDevice, cuda_status, sizeof(Grid), 1);
	}
	return cuda_status;
}

__host__ cudaError_t Grid::HostTransfer() { //NOTE
	cudaError_t cuda_status = cudaSuccess;
	cuda_status = CopyFunction("HostTransferParticlesPosition", particle_position_, particle_position_device_, cudaMemcpyDeviceToHost, cuda_status, sizeof(Vector3D), GetParticleCount());
	cuda_status = cudaDeviceSynchronize();

	return cuda_status;
}

__host__ __device__ size_t Grid::GetTotalSize() const {
#ifdef __CUDA_ARCH__
	return device_alloc_->total_size_;
#else
	return total_size_;
#endif
}

__host__ __device__ size_t Grid::GetParticleCount() const {
#ifdef __CUDA_ARCH__
	return device_alloc_->total_size_ * device_alloc_->resolution_;
#else
	return total_size_ * resolution_;
#endif
}

__host__ __device__ float Grid::GetResolution() const {
#ifdef __CUDA_ARCH__
	return device_alloc_->resolution_;
#else
	return resolution_;
#endif
}

__host__ __device__ Vector3D& Grid::GetVelocity(const size_t index) {
	assert(index <= GetParticleCount());
#ifdef __CUDA_ARCH__
	return particle_velocity_device_[index];
#else
	return particle_velocity_[index];
#endif
}

__host__ __device__ Vector3D& Grid::GetVelocity(IndexPair incident) {
	size_t index = incident.IX(side_size_);
	return GetVelocity(index);
}

__host__ __device__ Vector3D& Grid::GetPosition(const size_t index) {
	assert(index <= GetParticleCount());
#ifdef __CUDA_ARCH__
	return particle_position_device_[index];
#else
	return particle_position_[index];
#endif
}

__host__ __device__ Vector3D& Grid::GetPosition(IndexPair incident) {
	size_t index = incident.IX(side_size_);
	return GetPosition(index);
}

__host__ __device__ Matrix& Grid::GetMomentum(const size_t index) { //NOTE: DEVICE ONLY!
	assert(index <= GetParticleCount());
	return momentum_matrices_[index];
}

__host__ __device__ Matrix& Grid::GetMomentum(IndexPair incident) {
	size_t index = incident.IX(side_size_);
	return GetMomentum(index);
}

__host__ __device__ float& Grid::GetParticleMass(const size_t index) {
	assert(index <= GetParticleCount());
#ifdef __CUDA_ARCH__
	return particle_mass_device_[index];
#else
	return particle_mass_[index];
#endif
}

__host__ __device__ float& Grid::GetParticleMass(IndexPair incident) {
	size_t index = incident.IX(side_size_);
	return GetParticleMass(index);
}

__host__ __device__ float& Grid::GetCellMass(const size_t index) {
	assert(index <= total_size_);
#ifdef __CUDA_ARCH__
	return cell_mass_device_[index];
#else
	return cell_mass_[index];
#endif
}

__host__ __device__ float& Grid::GetCellMass(IndexPair incident) {
	size_t index = incident.IX(side_size_);
	return GetCellMass(index);
}

__host__ __device__ Vector3D& Grid::GetCellVelocity(const size_t index) {
	assert(index <= total_size_);
#ifdef __CUDA_ARCH__
	return cell_velocity_device_[index];
#else
	return cell_velocity_[index];
#endif
}

__host__ __device__ Vector3D& Grid::GetCellVelocity(IndexPair incident) {
	size_t index = incident.IX(side_size_);
	return GetCellVelocity(index);
}