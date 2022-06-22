#pragma once

#include "vector_field.hpp"
#include "cudamap.cuh"

#include <math.h>

enum class Direction { Origin, Left, Right, Up, Down };

struct FluidSim {
	/// <summary> Default Constructor </summary>
	FluidSim() = default;

	/// <summary> Loaded Constructor, allocates with all corresponding simulation arguments. </summary>
	FluidSim(float timestep, float diff, float visc, unsigned int size_x, unsigned int size_y, unsigned int iter, float time_max = 1);

	/// <summary> Adding density to the simulation at a specific IndexPair point. </summary>
	void AddDensity(IndexPair pair, float amount);

	/// <summary> Adding velocity to the simulation at a specific IndexPair point. </summary>
	void AddVelocity(IndexPair pair, float x, float y);

	/// <summary> Diffusion method for simulation. Should either be used with the viscocity or diffusion float in the "visc" argument. </summary>
	void Diffuse(int bounds, float visc, AxisData& current, AxisData& previous);

	/// <summary>  Projection method for simulation. Works with velocity exclusively. </summary>
	void Project(VectorField& v_current, VectorField& v_previous);

	/// <summary>  Advection method for simulation. </summary>
	void Advect(int bounds, AxisData& current, AxisData& previous, VectorField& velocity);

	/// <summary> Simulation method that combines uses defined simulation methods and then performs them to create a timestep. </summary>
	void Simulate();

	/// <summary> Allocates CUDAMaps with device memory. </summary>
	void AllocateDeviceData();

	/// <summary> Allocates CUDAMaps with device memory. </summary>
	void ReallocateHostData();

	void operator=(const FluidSim& copy);
	FluidSim& operator*();

	VectorField velocity_, velocity_prev_;
	AxisData density_, density_prev_;

	float diffusion_ = 0, viscosity_ = 0;

	float dt_ = 0, time_elapsed_ = 0, time_max_ = 0;

	unsigned int size_x_, size_y_; //Bounds
	unsigned int iterations_;

	cudaError_t cuda_status = cudaSuccess;

private:
	/// <summary> Gauss-Siedel systems of linear equations solver. /summary>
	void LinearSolve(int bounds, AxisData& current, AxisData& previous, float a_fac, float c_fac);
};