#pragma once

#include "vector_field.hpp"
#include <math.h>

#include <unordered_map>

using std::unordered_map;

struct FluidSim {
	FluidSim(float timestep, float diff, float visc, unsigned int size_x, unsigned int size_y, unsigned int iter);

	void AddDensity(IndexPair pair, float amount);
	void AddVelocity(IndexPair pair, float x, float y);

	void BoundaryConditions(int bounds, VectorField& input);

	VectorField Diffuse(int bounds, float visc, VectorField& current, VectorField& previous);
	void Project(VectorField& v_current, VectorField& v_previous);
	void Advect(int bounds, VectorField& current, VectorField& previous, VectorField& velocity);

	float dt_ = 0, diffusion_ = 0, viscosity_ = 0;
	unsigned int size_x_, size_y_; //Bounds
	unsigned int iterations_;
	VectorField density_, velocity_, velocity_prev_, density_prev_;

	enum class Direction { Origin, Left, Right, Up, Down };

private:
	void LinearSolve(int bounds, VectorField& current, VectorField& previous, float a_fac, float c_fac);
	unordered_map<Direction, IndexPair> GetAdjacentCoordinates(IndexPair incident); //Data Member
};