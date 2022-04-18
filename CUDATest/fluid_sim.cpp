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

VectorField FluidSim::Diffuse(int b, VectorField prev_field, float diff, float dt) {
	VectorField current_field = prev_field;
	float a = dt * diff * (iterations_ - 2) * (iterations_ - 2);

	LinearSolve(b, current_field, prev_field, a, 4 * a);
	return current_field;
}

void FluidSim::LinearSolve(int b, VectorField& current, VectorField previous, float a, float c) {
	unsigned int step, x, y, z,
		bound = size_x_ - 1; //z is for later when we add 3 dimensions

	for (step = 0; step < iterations_; step++) {
		for (x = 1; x < bound; x++) {
			for (y = 1; y < bound; y++) {
				map<IndexPair, F_Vector> c_map = current.GetVectorMap(),
					p_map = previous.GetVectorMap();

				map<FluidSim::Direction, IndexPair> pairs = GetAdjacentCoordinates(IndexPair(x, y));
				c_map[pairs[Direction::Origin]] = p_map[pairs[Direction::Origin]] + (c_map[pairs[Direction::Right]] * a)
					+ c_map[pairs[Direction::Left]]
					+ c_map[pairs[Direction::Up]]
					+ c_map[pairs[Direction::Down]]
					* (1.0 / c);

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