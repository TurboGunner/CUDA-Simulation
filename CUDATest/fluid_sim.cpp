#include "fluid_sim_cuda.cuh"
#include "fluid_sim.hpp"
#include "openvdb_handler.hpp"

#include <stdexcept>
#include <iostream>

FluidSim::FluidSim(float timestep, float diff, float visc, unsigned int size_x, unsigned int size_y, unsigned int iter, float time_max) {
	dt_ = timestep;
	diffusion_ = diff;
	viscosity_ = visc;
	size_x_ = size_x;
	size_y_ = size_y;

	if (iter == 0) {
		throw std::invalid_argument("Error: The number of iterations must be at least greater than or equal to 1!");
	}
	iterations_ = iter;
	time_max_ = time_max;

	velocity_ = VectorField(size_x, size_y);
	velocity_prev_ = VectorField(size_x, size_y);

	density_ = AxisData(size_x);
	density_prev_ = AxisData(size_x);
}

void FluidSim::AddDensity(IndexPair pair, float amount) {
	density_.map_->Put(pair, amount);;
}

void FluidSim::AddVelocity(IndexPair pair, float x, float y) {
	velocity_.GetVectorMap()[0].map_->Put(pair, x);
	velocity_.GetVectorMap()[1].map_->Put(pair, y);
}

void FluidSim::Diffuse(int bounds, float visc, AxisData& current, AxisData& previous) {
	float a = dt_ * visc * (size_x_ - 2) * (size_x_ - 2);

	LinearSolve(bounds, current, previous, a, 1 + 4 * a);
}

void FluidSim::DiffuseDensity(int bounds, float diff, AxisData& current, AxisData& previous) {
	float a = dt_ * diff * (size_x_ - 2) * (size_x_ - 2);
	LinearSolve(bounds, current, previous, a, 1 + 4 * a);
}

void FluidSim::Project(VectorField& v_current, VectorField& v_previous) {
	ProjectCuda(0, v_current, v_previous, size_x_, iterations_);
}

void FluidSim::Advect(int bounds, AxisData& current, AxisData& previous, VectorField& velocity) {
	AdvectCuda(0, current, previous, velocity, dt_, size_x_);
}

void FluidSim::LinearSolve(int bounds, AxisData& current, AxisData& previous, float a_fac, float c_fac) {
	LinearSolverCuda(bounds, current, previous, a_fac, c_fac, iterations_, size_x_);
}

void FluidSim::Simulate() {
	AddVelocity(IndexPair(5, 5), 120, 10);
	AddVelocity(IndexPair(1, 0), 222, 2);
	AddVelocity(IndexPair(32, 32), 22, 220);

	AddDensity(IndexPair(1, 1), 10.0f);
	AddDensity(IndexPair(2, 2), 100.0f);
	AddDensity(IndexPair(35, 35), 100.0f);

	OpenVDBHandler vdb_handler(*this);

	for (time_elapsed_ = 0; time_elapsed_ < time_max_ && time_elapsed_ <= 0; time_elapsed_ += dt_) { //Second bound condition is temporary!

		Diffuse(1, viscosity_, velocity_prev_.GetVectorMap()[0], velocity_.GetVectorMap()[0]);
		Diffuse(2, viscosity_, velocity_prev_.GetVectorMap()[1], velocity_.GetVectorMap()[1]);

		Project(velocity_prev_, velocity_);

		Advect(1, velocity_.GetVectorMap()[0], velocity_prev_.GetVectorMap()[0], velocity_);
		Advect(2, velocity_.GetVectorMap()[1], velocity_prev_.GetVectorMap()[1], velocity_);

		Project(velocity_, velocity_prev_);

		Diffuse(0, diffusion_, density_prev_, density_);
		Advect(0, density_, density_prev_, velocity_);

		vdb_handler.sim_ = *this;

		vdb_handler.WriteFile();
	}
}

void FluidSim::operator=(const FluidSim& copy) {
	density_ = copy.density_;
	density_prev_ = copy.density_prev_;

	velocity_ = copy.velocity_;
	velocity_prev_ = copy.velocity_prev_;

	size_x_ = copy.size_x_;
	size_y_ = copy.size_y_;

	dt_ = copy.dt_;
	diffusion_ = copy.diffusion_;
	viscosity_ = copy.viscosity_;
	iterations_ = copy.iterations_;
}

FluidSim& FluidSim::operator*() {
	return *this;
}