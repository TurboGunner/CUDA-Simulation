#pragma once

#include "vector_field.hpp"

#include <map>

using std::map;

struct FluidSim {
	FluidSim(float timestep, float diff, float visc, unsigned int size_x, unsigned int size_y, unsigned int iter);

	void AddDensity(IndexPair pair, float amountX, float amountY);
	void AddVelocity(IndexPair pair, float x, float y);

	VectorField Diffuse(int b, VectorField prev_field, float diff, float dt);

	float dt_ = 0, diffusion_ = 0, viscosity_ = 0;
	unsigned int size_x_ = 0, size_y_ = 0; //Bounds
	unsigned int iterations_;
	VectorField density_, velocity_;

	enum class Direction { Origin, Left, Right, Up, Down };

	private:
		void LinearSolve(int b, VectorField& current, VectorField previous, float a, float c);
		map<Direction, IndexPair> GetAdjacentCoordinates(IndexPair incident);
};