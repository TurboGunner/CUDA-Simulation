#pragma once

#include "vector_field.hpp"
#include <math.h>

#include <map>

using std::map;

struct FluidSim {
	FluidSim(float timestep, float diff, float visc, unsigned int size_x, unsigned int size_y, unsigned int iter);

	void AddDensity(IndexPair pair, float amount);
	void AddVelocity(IndexPair pair, float x, float y);

	void BoundaryConditions(int bounds, VectorField& input);

	VectorField Diffuse(int bounds, float diff, float dt);
	void Project();
	void Advect(int bounds, float dt);

	float dt_ = 0, diffusion_ = 0, viscosity_ = 0;
	unsigned int size_x_ = 0, size_y_ = 0; //Bounds
	unsigned int iterations_;
	VectorField density_, velocity_, density_prev_;

	enum class Direction { Origin, Left, Right, Up, Down };

	private:
		void LinearSolve(int bounds, VectorField& current, VectorField& previous, float a_fac, float c_fac);

		map<Direction, IndexPair> GetAdjacentCoordinates(IndexPair incident); //Data Member
};