#include "fluid_sim.hpp"

#include <stdexcept>
#include <iostream>

using std::map;

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

void FluidSim::AddDensity(IndexPair pair, float x, float y) {
	density_.GetVectorMap()[pair] = F_Vector(x, y);
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

			//std::cout << v_map[pairs[Direction::Right]].ToString() << std::endl;

			F_Vector calc =
				(v_map[pairs[Direction::Right]] - v_map[pairs[Direction::Left]])
				+ (v_map[pairs[Direction::Up]] - v_map[pairs[Direction::Down]]);
			calc = calc * -0.5f * (1.0f / size_x_);

			//std::cout << calc.ToString() << std::endl;

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
	VectorField& current = density_,
		previous = density_prev_,
		velocity = velocity_;

	unsigned int bound = size_x_ - 1;
	float x_current, x_previous, y_current, y_previous;

	float x_dt = dt * (size_x_ - 2);
	float y_dt = dt * (size_x_ - 2);

	float velocity_x_curr, velocity_x_prev, velocity_y_curr, velocity_y_prev;
	float x, y;

	int x_value, y_value;

	map<IndexPair, F_Vector> c_map = current.GetVectorMap(),
		p_map = previous.GetVectorMap(),
		v_map = velocity.GetVectorMap();

	for (y = 1; y < bound; y++) {
		for (x = 1; x < bound; x++) {
			x_value = x - (x_dt * v_map[IndexPair(x, y)].vx);
			y_value = y - (y_dt * v_map[IndexPair(x, y)].vy);

			if (x_value < 0.5f) {
				x_value = 0.5f;
			}
			if (x_value > size_x_ + 0.5f) {
				x_value = size_x_ + 0.5f;
			}
			x_current = x_value;
			x_previous = x_current + 1.0f;
			if (y_value < 0.5f) {
				y_value = 0.5f;
			}
			if (y_value > size_x_ + 0.5f) {
				y_value = size_x_ + 0.5f;
			}
			y_current = y_value;
			y_previous = y_current + 1.0f;

			velocity_x_prev = x_value - x_current;
			velocity_x_curr = 1.0f - velocity_x_prev;
			velocity_y_prev = y_value - y_current;
			velocity_y_curr = 1.0f - velocity_y_prev;

			c_map[IndexPair(x, y)] =
				((p_map[IndexPair(int(x_current), int(y_current))] * velocity_y_curr) +
				(p_map[IndexPair(int(x_current), int(y_previous))] * velocity_y_prev) * velocity_x_curr) +
				((p_map[IndexPair(int(x_previous), int(y_current))] * velocity_y_curr) +
				(p_map[IndexPair(int(x_previous), int(y_previous))] * velocity_y_prev) * velocity_x_prev);
		}
	}
	BoundaryConditions(bounds, current);
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
	unsigned int step, x, y, z,
		bound = size_x_ - 1; //z is for later when we add 3 dimensions
	map<IndexPair, F_Vector>& c_map = current.GetVectorMap(),
		p_map = previous.GetVectorMap();

	unsigned int array_size = bound * bound;

	for (step = 0; step < iterations_; step++) {
		for (y = 1; y < bound; y++) {
			for (x = 1; x < bound; x++) {

				map<FluidSim::Direction, IndexPair> pairs = GetAdjacentCoordinates(IndexPair(x, y));

				F_Vector calc = c_map[pairs[Direction::Right]]
					+ c_map[pairs[Direction::Left]]
					+ c_map[pairs[Direction::Up]]
					+ c_map[pairs[Direction::Down]];
				calc = calc * a_fac;

				calc = calc + p_map[pairs[Direction::Origin]];
				c_map[pairs[Direction::Origin]] = calc * (1.0f / c_fac);
			}
		}
	}
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