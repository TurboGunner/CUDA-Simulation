#include "fluid_sim_cuda.cuh"
#include "fluid_sim.hpp"

#include <stdexcept>
#include <iostream>

FluidSim::FluidSim(float timestep, float diff, float visc, unsigned int size_x, unsigned int size_y, unsigned int iter) {
	dt_ = timestep;
	diffusion_ = diff;
	viscosity_ = visc;
	size_x_ = size_x;
	size_y_ = size_y;

	if (iter == 0) {
		throw std::invalid_argument("Error: The number of iterations must be at least greater than or equal to 1!");
	}
	iterations_ = iter;

	density_ = VectorField(size_x, size_y);
	velocity_ = VectorField(size_x, size_y);
	density_prev_ = VectorField(size_x, size_y);
}

void FluidSim::AddDensity(IndexPair pair, float amount) {
	density_.GetVectorMap()[pair] = F_Vector(amount, amount);
}

void FluidSim::AddVelocity(IndexPair pair, float x, float y) {
	velocity_.GetVectorMap()[pair] = F_Vector(x, y);
}

VectorField FluidSim::Diffuse(int bounds, float visc, float dt, VectorField& current, VectorField& previous) {
	float a = dt * visc * (size_x_ - 2) * (size_x_ - 2);

	LinearSolve(bounds, current, previous, a, 1 + 4 * a);
	std::cout << density_.ToString() << std::endl;
	return current;
}

void FluidSim::Project(VectorField& v_current, VectorField& v_previous) {
	ProjectCuda(0, v_current, v_previous, size_x_, iterations_);
}

void FluidSim::Advect(int bounds, float dt) {
	AdvectCuda(0, density_, density_prev_, velocity_, dt, size_x_);
	std::cout << density_.ToString() << std::endl;
}

void FluidSim::BoundaryConditions(int bounds, VectorField& input) {
	unsigned int bound = size_x_ - 1;
	unordered_map<IndexPair, F_Vector, Hash>& c_map = input.GetVectorMap();

	for (int i = 1; i < bound; i++) {
		c_map[IndexPair(i, 0)].vx = bounds == 1 ? c_map[IndexPair(i, 0)].vx * -1.0f : c_map[IndexPair(i, 1)].vx;
		c_map[IndexPair(i, bound)].vx = bounds == 1 ? c_map[IndexPair(i, bound - 1)].vx * -1.0f : c_map[IndexPair(i, bound - 1)].vx;
	}
	for (int j = 1; j < bound; j++) {
		c_map[IndexPair(0, j)].vy = bounds == 1 ? c_map[IndexPair(1, j)].vy * -1.0f : c_map[IndexPair(1, j)].vy;
		c_map[IndexPair(bound, j)] = bounds == 1 ? c_map[IndexPair(bound - 1, j)] * -1.0f : c_map[IndexPair(bound - 1, j)].vy;
	}

	c_map[IndexPair(0, 0)] = c_map[IndexPair(1, 0)] + c_map[IndexPair(0, 1)] * .5f;
	c_map[IndexPair(0, bound)] = c_map[IndexPair(1, bound)] + c_map[IndexPair(0, bound - 1)] * .5f;
	c_map[IndexPair(bound, 0)] = c_map[IndexPair(bound - 1, 0)] + c_map[IndexPair(bound, 1)] * .5f;
	c_map[IndexPair(bound, bound)] = c_map[IndexPair(bound - 1, bound)] + c_map[IndexPair(bound, bound - 1)] * .5f;
}

void FluidSim::LinearSolve(int bounds, VectorField& current, VectorField& previous, float a_fac, float c_fac) {
	LinearSolverCuda(bounds, current, previous, a_fac, c_fac, iterations_, size_x_);
	BoundaryConditions(bounds, current);
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