#include "fluid_sim_cuda.cuh"
#include "fluid_sim.hpp"
#include "openvdb_handler.hpp"

#include <stdexcept>
#include <iostream>

FluidSim::FluidSim(float timestep, float diff, float visc, uint3 size, unsigned int iter, float time_max) {
	dt_ = timestep;
	diffusion_ = diff;
	viscosity_ = visc;
	size_ = size;

	if (iter == 0) {
		throw std::invalid_argument("Error: The number of iterations must be at least greater than or equal to 1!");
	}
	iterations_ = iter;
	time_max_ = time_max;

	velocity_ = VectorField(size_);
	velocity_prev_ = VectorField(size_);

	density_ = AxisData(size_);
	density_prev_ = AxisData(size_);
}

void FluidSim::AddDensity(IndexPair pair, float amount) {
	if (pair.x >= size_.y || pair.y >= size_.y || pair.z >= size_.z) {
		throw std::invalid_argument("Error: The IndexPair arguments for the fluid simulation are out of bounds!");
	}

	density_add_total_ += amount;

	AddOnAxisCuda(density_, pair, amount, size_);
}

void FluidSim::AddVelocity(IndexPair pair, float x, float y, float z) {
	if (pair.x >= size_.y || pair.y >= size_.y || pair.z >= size_.z) {
		throw std::invalid_argument("Error: The IndexPair arguments for the fluid simulation are out of bounds!");
	}
	float3 float_vec;

	float_vec.x = x;
	float_vec.y = y;
	float_vec.z = z;

	v_add_total_.x += x;
	v_add_total_.y += y;
	v_add_total_.z += z;

	AddOnVectorCuda(velocity_, pair, float_vec, size_);
}

void FluidSim::Diffuse(int bounds, float visc, AxisData& current, AxisData& previous) {
	float a = dt_ * visc * (size_.x - 2) * (size_.y - 2);

	LinearSolve(bounds, current, previous, a, 1 + 6 * a);
}

void FluidSim::Project(VectorField& v_current, VectorField& v_previous) {
	ProjectCuda(0, v_current, v_previous, size_, iterations_);
}

void FluidSim::Advect(int bounds, AxisData& current, AxisData& previous, VectorField& velocity) {
	AdvectCuda(bounds, current, previous, velocity, dt_, size_);
}

void FluidSim::LinearSolve(int bounds, AxisData& current, AxisData& previous, float a_fac, float c_fac) {
	LinearSolverCuda(bounds, current, previous, a_fac, c_fac, iterations_, size_);
}

void FluidSim::VelocityStep() {
	Diffuse(1, viscosity_, velocity_prev_.map_[0], velocity_.map_[0]);
	Diffuse(2, viscosity_, velocity_prev_.map_[1], velocity_.map_[1]);
	Diffuse(3, viscosity_, velocity_prev_.map_[2], velocity_.map_[2]);

	Project(velocity_prev_, velocity_);

	Advect(1, velocity_.map_[0], velocity_prev_.map_[0], velocity_prev_);
	Advect(2, velocity_.map_[1], velocity_prev_.map_[1], velocity_prev_);
	Advect(3, velocity_.map_[2], velocity_prev_.map_[2], velocity_prev_);

	Project(velocity_, velocity_prev_);
}

void FluidSim::DensityStep() {
	Diffuse(0, diffusion_, density_prev_, density_);

	Advect(0, density_, density_prev_, velocity_);
}

void FluidSim::Simulate() {

	AllocateDeviceData();

	AddVelocity(IndexPair(62, 62, 62), 5.0f, 5.0f, 0);

	AddDensity(IndexPair(62, 62, 62), 10.0f);

	OpenVDBHandler vdb_handler(*this);

	for (time_elapsed_ = 0; time_elapsed_ < time_max_; time_elapsed_ += dt_) { //Second bound condition is temporary!

		VelocityStep();

		DensityStep();

		ReallocateHostData();

		AddVelocity(IndexPair(62, 62, 62), 25.0f, 25.0f, 0);
		AddDensity(IndexPair(62, 62, 62), 100.0f);

		vdb_handler.WriteFile(velocity_.map_[0], velocity_.map_[1], velocity_.map_[2], density_);

		std::cout << "Density: " << density_.map_->Get(IndexPair(61, 61, 61).IX(size_.x)) << std::endl;
		std::cout << "Velocity: " << velocity_.map_[0].map_->Get(IndexPair(61, 61, 61).IX(size_.x)) << std::endl;
	}
}

void FluidSim::AllocateDeviceData() {
	density_.map_->DeviceTransfer(cuda_status, density_.map_, d_map);
	density_prev_.map_->DeviceTransfer(cuda_status, density_prev_.map_, d_prev_map);

	velocity_.map_[0].map_->DeviceTransfer(cuda_status, velocity_.map_[0].map_, v_map_x);
	velocity_.map_[1].map_->DeviceTransfer(cuda_status, velocity_.map_[1].map_, v_map_y);
	velocity_.map_[2].map_->DeviceTransfer(cuda_status, velocity_.map_[2].map_, v_map_z);

	velocity_prev_.map_[0].map_->DeviceTransfer(cuda_status, velocity_prev_.map_[0].map_, v_prev_map_x);
	velocity_prev_.map_[1].map_->DeviceTransfer(cuda_status, velocity_prev_.map_[1].map_, v_prev_map_y);
	velocity_prev_.map_[2].map_->DeviceTransfer(cuda_status, velocity_prev_.map_[2].map_, v_prev_map_z);
}

void FluidSim::ReallocateHostData() {
	density_.map_->HostTransfer(cuda_status);
	density_prev_.map_->HostTransfer(cuda_status);

	velocity_.map_[0].map_->HostTransfer(cuda_status);
	velocity_.map_[1].map_->HostTransfer(cuda_status);
	velocity_.map_[2].map_->HostTransfer(cuda_status);

	velocity_prev_.map_[0].map_->HostTransfer(cuda_status);
	velocity_prev_.map_[1].map_->HostTransfer(cuda_status);
	velocity_prev_.map_[2].map_->HostTransfer(cuda_status);
	cudaDeviceSynchronize();
}

void FluidSim::operator=(const FluidSim& copy) {
	density_ = copy.density_;
	density_prev_ = copy.density_prev_;

	velocity_ = copy.velocity_;
	velocity_prev_ = copy.velocity_prev_;

	size_ = copy.size_;

	dt_ = copy.dt_;
	diffusion_ = copy.diffusion_;
	viscosity_ = copy.viscosity_;
	iterations_ = copy.iterations_;
}

FluidSim& FluidSim::operator*() {
	return *this;
}