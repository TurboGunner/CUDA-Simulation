#pragma once

#include "cuda_runtime.h"

#include "handler_methods.hpp"

#include <string>
#include <random>

using std::string;

struct F_Vector {
	/// <summary> 
	/// Default constructor. Has defaults for all dimension sizes as 1.
	/// </summary>
	__host__ __device__ F_Vector(float x_in = 1, float y_in = 1);

	/// <summary> 
	/// Returns a float pointer of all F_Vector values.
	/// </summary>
	float* AsArrPair();

	/// <summary> 
	/// Gets the magnitude of the vector using the provided references values.
	/// </summary>
	float Magnitude() const;

	/// <summary> 
	/// Operator overload for equality of two vectors.
	/// </summary>
	bool operator==(const F_Vector& v1) const;

	/// <summary> 
	/// Operator overload for addition of two vectors elementwise.
	/// </summary>
	__host__ __device__ const F_Vector operator+(const F_Vector& v1) const;

	/// <summary> 
	/// Operator overload for subtraction of two vectors elementwise.
	/// </summary>
	__host__ __device__ const F_Vector operator-(const F_Vector& v1) const;

	/// <summary> 
	/// Operator overload for multiplication of a vector by a provided float elementwise.
	/// </summary>
	__host__ __device__ F_Vector operator*(float num);

	/// <summary> 
	/// Operator overload for multiplication of a vector by a provided unsigned int elementwise.
	/// </summary>
	__host__ __device__ F_Vector operator*(unsigned int num);

	/// <summary> 
	/// Operator overload for returning the proper hash code for F_Vector.
	/// </summary>
	size_t operator()(const F_Vector& v1) const noexcept;

	/// <summary> 
	/// Operator overload for copying the data of an existing F_Vector.
	/// </summary>
	__host__ __device__ void operator=(const F_Vector& copy);

	/// <summary> 
	/// Returns an std::string of the components of F_Vector.
	/// </summary>
	string ToString() const;

	float vx_, vy_; //Components
	float *vx_ptr_, *vy_ptr_;
};

/// <summary> 
/// Generates a vector with random values based on the Mersenne Twister algorithm.
/// </summary>
inline F_Vector RandomVector() {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(1, 10);

	F_Vector f = F_Vector(dist(gen), dist(gen));
	return f;
}