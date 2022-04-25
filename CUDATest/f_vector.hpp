#pragma once

#include <string>

using std::string;

struct F_Vector {
	/// <summary> 
	/// Default constructor. Has defaults for all dimension sizes as 1.
	/// </summary>
	F_Vector(float x_in = 1, float y_in = 1);

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
	const F_Vector operator+(const F_Vector& v1) const;

	/// <summary> 
	/// Operator overload for subtraction of two vectors elementwise.
	/// </summary>
	const F_Vector operator-(const F_Vector& v1) const;

	/// <summary> 
	/// Operator overload for multiplication of a vector by a provided float elementwise.
	/// </summary>
	F_Vector operator*(float num);

	/// <summary> 
	/// Operator overload for multiplication of a vector by a provided unsigned int elementwise.
	/// </summary>
	F_Vector operator*(unsigned int num);

	/// <summary> 
	/// Operator overload for returning the proper hash code for F_Vector.
	/// </summary>
	size_t operator()(const F_Vector& v1) const noexcept;

	/// <summary> 
	/// Operator overload for copying the data of an existing F_Vector.
	/// </summary>
	void operator=(const F_Vector& copy);

	/// <summary> 
	/// Returns an std::string of the components of F_Vector.
	/// </summary>
	string ToString() const;

	float vx, vy; //Components
};