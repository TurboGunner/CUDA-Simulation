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

VectorField FluidSim::Diffuse(int bounds, float diff, float dt) {
	VectorField& current = density_;
	VectorField& previous = density_prev_;
	float a = dt * diff * (size_x_ - 2) * (size_x_ - 2);

	LinearSolve(bounds, current, previous, a, 1 + 4 * a);
	return current;
}

void FluidSim::Project() {

	unsigned int x, y, bound = size_x_ - 1;
	map<IndexPair, F_Vector>& c_map = density_.GetVectorMap(),
		p_map = density_prev_.GetVectorMap(),
		v_map = velocity_.GetVectorMap();

	for (y = 1; y < bound; y++) {
		for (x = 1; x < bound; x++) {

			map<FluidSim::Direction, IndexPair> pairs = GetAdjacentCoordinates(IndexPair(x, y));

			F_Vector calc =
				(v_map[pairs[Direction::Right]] - v_map[pairs[Direction::Left]])
				+ (v_map[pairs[Direction::Up]] - v_map[pairs[Direction::Down]]);
			calc = calc * -0.5f * (1.0f / size_x_);

			c_map[IndexPair(x, y)] = calc;
			p_map[pairs[Direction::Origin]] = 0;
		}
	}
	BoundaryConditions(0, density_);
	BoundaryConditions(0, density_prev_);
	LinearSolve(0, density_, density_prev_, 1, 4);

	for (y = 1; y < bound; y++) {
		for (x = 1; x < bound; x++) {
			map<FluidSim::Direction, IndexPair> pairs = GetAdjacentCoordinates(IndexPair(x, y));
			v_map[pairs[Direction::Origin]].vx -= 0.5f * v_map[pairs[Direction::Right]].vx
				- v_map[pairs[Direction::Left]].vx * size_x_;
			v_map[pairs[Direction::Origin]].vy -= 0.5f * v_map[pairs[Direction::Up]].vy
				- v_map[pairs[Direction::Down]].vy * size_x_;
		}
	}
	BoundaryConditions(1, velocity_);
}

void FluidSim::Advect(int bounds, float dt) {

	float* result_ptr = AdvectCuda(0, density_, density_prev_, velocity_, dt, size_x_);
}

void FluidSim::BoundaryConditions(int bounds, VectorField& input) {
	unsigned int bound = size_x_ - 1;
	map<IndexPair, F_Vector>& c_map = input.GetVectorMap();

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
	c_map[IndexPair(bound, 0)] = c_map[IndexPair(bound - 1, 0)] +  c_map[IndexPair(bound, 1)] * .5f;
	c_map[IndexPair(bound, bound)] = c_map[IndexPair(bound - 1, bound)] + c_map[IndexPair(bound, bound - 1)] * .5f;
}

void FluidSim::LinearSolve(int bounds, VectorField& current, VectorField& previous, float a_fac, float c_fac) {
	float* results_x = LinearSolverCuda(0, current, previous, a_fac, c_fac, iterations_, size_x_);

	current.RepackMap(results_x, results_x);

	free(results_x);

	BoundaryConditions(bounds, current);
}

map<FluidSim::Direction, IndexPair> FluidSim::GetAdjacentCoordinates(IndexPair incident) {
	map<FluidSim::Direction, IndexPair> output;
	output.emplace(Direction::Origin, incident);

	output.emplace(Direction::Left, IndexPair(incident.x - 1, incident.y));
	output.emplace(Direction::Right, IndexPair(incident.x + 1, incident.y));

	output.emplace(Direction::Up, IndexPair(incident.x, incident.y + 1));
	output.emplace(Direction::Down, IndexPair(incident.x, incident.y - 1));

	return output;
}