#pragma once

#include "vector_field.hpp"
#include "cudamap.cuh"

#include <math.h>

struct CoordinateHash {
	__host__ __device__ size_t operator()(const FluidSim::Direction& d1) const {
		size_t base_hash = (size_t)d1;
		return base_hash ^ (base_hash << 1);
	}
};

struct FluidSim {
	FluidSim() = default;
	FluidSim(float timestep, float diff, float visc, unsigned int size_x, unsigned int size_y, unsigned int iter, float time_max = 1);

	void AddDensity(IndexPair pair, float amount);
	void AddVelocity(IndexPair pair, float x, float y);

	void Diffuse(int bounds, float visc, AxisData& current, AxisData& previous);
	void DiffuseDensity(int bounds, float diff, AxisData& current, AxisData& previous);

	void Project(VectorField& v_current, VectorField& v_previous);
	void Advect(int bounds, AxisData& current, AxisData& previous, VectorField& velocity);
	void Simulate();

	VectorField velocity_, velocity_prev_;
	AxisData density_, density_prev_;

	float dt_ = 0, diffusion_ = 0, viscosity_ = 0;

	unsigned int size_x_, size_y_; //Bounds
	unsigned int iterations_;

	float time_elapsed_ = 0;
	float time_max_ = 0;

	void operator=(const FluidSim& copy);
	FluidSim& operator*();

	enum class Direction { Origin, Left, Right, Up, Down };

private:
	void LinearSolve(int bounds, AxisData& current, AxisData& previous, float a_fac, float c_fac);
};

__host__ __device__ inline HashMap<FluidSim::Direction, IndexPair, CoordinateHash>* GetAdjacentCoordinates(IndexPair incident, unsigned int size) {
	auto* output = new HashMap<FluidSim::Direction, IndexPair, CoordinateHash>(size);
	output->Put(FluidSim::Direction::Origin, incident);

	output->Put(FluidSim::Direction::Left, IndexPair(incident.x - 1, incident.y));
	output->Put(FluidSim::Direction::Right, IndexPair(incident.x + 1, incident.y));

	output->Put(FluidSim::Direction::Up, IndexPair(incident.x, incident.y + 1));
	output->Put(FluidSim::Direction::Down, IndexPair(incident.x, incident.y - 1));

	return output;
}