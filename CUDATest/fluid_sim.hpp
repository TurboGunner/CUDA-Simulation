#pragma once

#include "vector_field.hpp"
#include "cudamap.cuh"

#include <math.h>

enum class Direction { Origin, Left, Right, Up, Down };

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

private:
	void LinearSolve(int bounds, AxisData& current, AxisData& previous, float a_fac, float c_fac);
};

__host__ __device__ inline HashMap<Direction, IndexPair, HashFunc<Direction>>* GetAdjacentCoordinates(IndexPair incident) {
	auto* output = new HashMap<Direction, IndexPair, HashFunc<Direction>>(5);
	output->Put(Direction::Origin, incident);

	output->Put(Direction::Left, IndexPair(incident.x - 1, incident.y));
	output->Put(Direction::Right, IndexPair(incident.x + 1, incident.y));

	output->Put(Direction::Up, IndexPair(incident.x, incident.y + 1));
	output->Put(Direction::Down, IndexPair(incident.x, incident.y - 1));

	return output;
}

__host__ __device__ inline void BoundaryConditions(int bounds, HashMap<IndexPair, F_Vector, Hash<IndexPair>>* c_map, int side_size) {
	unsigned int bound = side_size - 1;

	for (int i = 1; i < bound; i++) {
		(*c_map)[IndexPair(i, 0)].vx_ = bounds == 1 ? (*c_map)[IndexPair(i, 0)].vx_ * -1.0f : (*c_map)[IndexPair(i, 1)].vx_;
		(*c_map)[IndexPair(i, bound)].vx_ = bounds == 1 ? (*c_map)[IndexPair(i, bound - 1)].vx_ * -1.0f : (*c_map)[IndexPair(i, bound - 1)].vx_;
	}
	for (int j = 1; j < bound; j++) {
		(*c_map)[IndexPair(0, j)].vy_ = bounds == 1 ? (*c_map)[IndexPair(1, j)].vy_ * -1.0f : (*c_map)[IndexPair(1, j)].vy_;
		(*c_map)[IndexPair(bound, j)] = bounds == 1 ? (*c_map)[IndexPair(bound - 1, j)] * -1.0f : (*c_map)[IndexPair(bound - 1, j)].vy_;
	}

	(*c_map)[IndexPair(0, 0)] = (*c_map)[IndexPair(1, 0)] + (*c_map)[IndexPair(0, 1)] * .5f;
	(*c_map)[IndexPair(0, bound)] = (*c_map)[IndexPair(1, bound)] + (*c_map)[IndexPair(0, bound - 1)] * .5f;
	(*c_map)[IndexPair(bound, 0)] = (*c_map)[IndexPair(bound - 1, 0)] + (*c_map)[IndexPair(bound, 1)] * .5f;
	(*c_map)[IndexPair(bound, bound)] = (*c_map)[IndexPair(bound - 1, bound)] + (*c_map)[IndexPair(bound, bound - 1)] * .5f;
}

__host__ __device__ inline void BoundaryConditions(int bounds, HashMap<IndexPair, float, HashDupe<IndexPair>>* c_map, int side_size) {
	unsigned int bound = side_size - 1;

	for (int i = 1; i < bound; i++) {
		(*c_map)[IndexPair(i, 0)] = bounds == 2 ? (*c_map)[IndexPair(i, 0)] * -1.0f : (*c_map)[IndexPair(i, 1)];
		(*c_map)[IndexPair(i, bound)] = bounds == 2 ? (*c_map)[IndexPair(i, bound - 1)] * -1.0f : (*c_map)[IndexPair(i, bound - 1)];
	}
	for (int j = 1; j < bound; j++) {
		(*c_map)[IndexPair(0, j)] = bounds == 1 ? (*c_map)[IndexPair(1, j)] * -1.0f : (*c_map)[IndexPair(1, j)];
		(*c_map)[IndexPair(bound, j)] = bounds == 1 ? (*c_map)[IndexPair(bound - 1, j)] * -1.0f : (*c_map)[IndexPair(bound - 1, j)];
	}

	(*c_map)[IndexPair(0, 0)] = (*c_map)[IndexPair(1, 0)] + (*c_map)[IndexPair(0, 1)] * .5f;
	(*c_map)[IndexPair(0, bound)] = (*c_map)[IndexPair(1, bound)] + (*c_map)[IndexPair(0, bound - 1)] * .5f;
	(*c_map)[IndexPair(bound, 0)] = (*c_map)[IndexPair(bound - 1, 0)] + (*c_map)[IndexPair(bound, 1)] * .5f;
	(*c_map)[IndexPair(bound, bound)] = (*c_map)[IndexPair(bound - 1, bound)] + (*c_map)[IndexPair(bound, bound - 1)] * .5f;
}