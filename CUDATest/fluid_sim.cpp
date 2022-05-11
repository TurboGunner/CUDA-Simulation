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
	velocity_prev_ = velocity_;

	density_ = AxisData(size_x);
	density_prev_ = density_;
}

void FluidSim::AddDensity(IndexPair pair, float amount) {
	density_.map_[pair] = amount;
}

void FluidSim::AddVelocity(IndexPair pair, float x, float y) {
	velocity_.GetVectorMap()[pair] = F_Vector(x, y);
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

void FluidSim::BoundaryConditions(int bounds, VectorField& input) {
	unsigned int bound = size_x_ - 1;
	unordered_map<IndexPair, F_Vector, Hash>& c_map = input.GetVectorMap();

	for (int i = 1; i < bound; i++) {
		c_map[IndexPair(i, 0)].vx_ = bounds == 1 ? c_map[IndexPair(i, 0)].vx_ * -1.0f : c_map[IndexPair(i, 1)].vx_;
		c_map[IndexPair(i, bound)].vx_ = bounds == 1 ? c_map[IndexPair(i, bound - 1)].vx_ * -1.0f : c_map[IndexPair(i, bound - 1)].vx_;
	}
	for (int j = 1; j < bound; j++) {
		c_map[IndexPair(0, j)].vy_ = bounds == 1 ? c_map[IndexPair(1, j)].vy_ * -1.0f : c_map[IndexPair(1, j)].vy_;
		c_map[IndexPair(bound, j)] = bounds == 1 ? c_map[IndexPair(bound - 1, j)] * -1.0f : c_map[IndexPair(bound - 1, j)].vy_;
	}

	c_map[IndexPair(0, 0)] = c_map[IndexPair(1, 0)] + c_map[IndexPair(0, 1)] * .5f;
	c_map[IndexPair(0, bound)] = c_map[IndexPair(1, bound)] + c_map[IndexPair(0, bound - 1)] * .5f;
	c_map[IndexPair(bound, 0)] = c_map[IndexPair(bound - 1, 0)] + c_map[IndexPair(bound, 1)] * .5f;
	c_map[IndexPair(bound, bound)] = c_map[IndexPair(bound - 1, bound)] + c_map[IndexPair(bound, bound - 1)] * .5f;
}

void FluidSim::LinearSolve(int bounds, AxisData& current, AxisData& previous, float a_fac, float c_fac) {
	LinearSolverCuda(bounds, current, previous, a_fac, c_fac, iterations_, size_x_);
}

unordered_map<FluidSim::Direction, IndexPair> FluidSim::GetAdjacentCoordinates(IndexPair incident) {
	unordered_map<FluidSim::Direction, IndexPair> output;
	output.emplace(Direction::Origin, incident);

	output.emplace(Direction::Left, IndexPair(incident.x - 1, incident.y));
	output.emplace(Direction::Right, IndexPair(incident.x + 1, incident.y));

	output.emplace(Direction::Up, IndexPair(incident.x, incident.y + 1));
	output.emplace(Direction::Down, IndexPair(incident.x, incident.y - 1));

	return output;
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
		AxisData v_prev_x, v_x, v_prev_y, v_y;

		velocity_.DataConstrained(Axis::X, v_x);
		velocity_prev_.DataConstrained(Axis::X, v_prev_x);
		velocity_.DataConstrained(Axis::Y, v_y);
		velocity_prev_.DataConstrained(Axis::Y, v_prev_y);

		Diffuse(1, viscosity_, v_prev_x, v_x);
		velocity_.RepackFromConstrained(v_x);
		velocity_prev_.RepackFromConstrained(v_prev_x);

		Diffuse(2, viscosity_, v_prev_y, v_y);
		velocity_.RepackFromConstrained(v_y);
		velocity_prev_.RepackFromConstrained(v_prev_y);

		Project(velocity_prev_, velocity_);

		velocity_.DataConstrained(Axis::X, v_x);
		velocity_prev_.DataConstrained(Axis::X, v_prev_x);
		Advect(1, v_x, v_prev_x, velocity_);

		velocity_.DataConstrained(Axis::Y, v_y);
		velocity_prev_.DataConstrained(Axis::Y, v_prev_y);
		Advect(2, v_y, v_prev_y, velocity_);

		Project(velocity_, velocity_prev_);
		Diffuse(0, diffusion_, density_prev_, density_);
		Advect(0, density_prev_, density_, velocity_);

		vdb_handler.sim_ = *this;

		vdb_handler.WriteFile();

	}

	//std::cout << simulation.velocity_.ToString() << std::endl;
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