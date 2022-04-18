#include "fluid_sim.hpp"

#include <stdexcept>

using std::map;

FluidSim::FluidSim(float timestep, float diff, float visc, unsigned int size_x, unsigned int size_y, unsigned int iter) {
	dt_ = timestep;
	diffusion_ = diff;
	viscosity_ = visc;
	size_x_ = size_x;
	size_y_ = size_y;

	if (iter == 0) {
		throw std::invalid_argument("Error: Bounds must be at least greater than or equal to 1!");
	}
	iterations_ = iter;

	density_ = VectorField(size_x, size_y);
	velocity_ = VectorField(size_x, size_y);
}

void FluidSim::AddDensity(IndexPair pair, float x, float y) {
	density_.GetVectorMap()[pair] = F_Vector(x, y);
}

void FluidSim::AddVelocity(IndexPair pair, float x, float y) {
	velocity_.GetVectorMap()[pair] = F_Vector(x, y);
}

VectorField FluidSim::Diffuse(int b, VectorField& current, VectorField& previous, float diff, float dt) {
	float a = dt * diff * (iterations_ - 2) * (iterations_ - 2);

	LinearSolve(b, current, previous, a, 4 * a);
	return current;
}

void FluidSim::Project(VectorField& current, VectorField& previous, VectorField& velocity) {
	unsigned int x, y, bound = size_x_ - 1;

	for (y = 1; y < bound; y++) {
		for (x = 1; x < bound; x++) {

			map<IndexPair, F_Vector> c_map = current.GetVectorMap(),
				p_map = previous.GetVectorMap(),
				v_map = velocity.GetVectorMap();

			map<FluidSim::Direction, IndexPair> pairs = GetAdjacentCoordinates(IndexPair(x, y));

			F_Vector calc =
				(v_map[pairs[Direction::Right]] - v_map[pairs[Direction::Left]]
				+ v_map[pairs[Direction::Up]] - v_map[pairs[Direction::Down]]
				);
			calc = calc * (1 / iterations_) * -0.5f;

			c_map[IndexPair(x, y)] = calc;
			p_map[pairs[Direction::Origin]] = 0;
		}
	}
	LinearSolve(0, current, previous, 1, 6);
}

void FluidSim::Advect(int b, VectorField& current, VectorField& previous, VectorField& velocity, float dt) {
	unsigned int bound = size_x_ - 1;
	float x_current, x_previous, y_current, y_previous;

	float x_dt = dt * (iterations_ - 2);
	float y_dt = dt * (iterations_ - 2);

	float s0, s1, t0, t1;
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
			if (x_value > iterations_ + 0.5f) {
				x_value = iterations_ + 0.5f;
			}
			x_current = floor(x_value);
			x_previous = x_current + 1.0f;
			if (y_value < 0.5f) {
				y_value = 0.5f;
			}
			if (y_value > iterations_ + 0.5f) {
				y_value = iterations_ + 0.5f;
			}
			y_current = floor(y_value);
			x_previous = x_current + 1.0f;

			s1 = x_value - x_current;
			s0 = 1.0f - s1;
			t1 = y_value - y_current;
			t0 = 1.0f - t1;

			c_map[IndexPair(x, y)] =
				((p_map[IndexPair(int(x_current), int(y_current))] * t0) +
				(p_map[IndexPair(int(x_current), int(y_previous))] * t1) * s0) +
				((p_map[IndexPair(int(x_previous), int(y_current))] * t0) +
				(p_map[IndexPair(int(x_previous), int(y_previous))] * t1) * s1);
		}
	}
}

void FluidSim::LinearSolve(int b, VectorField& current, VectorField& previous, float a_fac, float c_fac) {
	unsigned int step, x, y, z,
		bound = size_x_ - 1; //z is for later when we add 3 dimensions

	for (step = 0; step < iterations_; step++) {
		for (x = 1; x < bound; x++) {
			for (y = 1; y < bound; y++) {
				map<IndexPair, F_Vector> c_map = current.GetVectorMap(),
					p_map = previous.GetVectorMap();

				map<FluidSim::Direction, IndexPair> pairs = GetAdjacentCoordinates(IndexPair(x, y));

				c_map[pairs[Direction::Origin]] = p_map[pairs[Direction::Origin]]
					+ (c_map[pairs[Direction::Right]] * a_fac)
					+ c_map[pairs[Direction::Left]]
					+ c_map[pairs[Direction::Up]]
					+ c_map[pairs[Direction::Down]]
					* (1.0f / c_fac);
			}
		}
	}
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