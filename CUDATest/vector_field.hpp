#pragma once

#include "cuda_runtime.h"
#include <string>
#include <set>
#include <map>

using std::string;
using std::set;
using std::map;

struct F_Vector {
	F_Vector(float x_in = 1, float y_in = 1);

	float* AsArrPair();

	float Magnitude() const;

	bool operator==(const F_Vector& v1) const;

	const F_Vector operator+(const F_Vector& v1) const;
	const F_Vector operator-(const F_Vector& v1) const;

	F_Vector operator*(float num);
	F_Vector operator*(unsigned int num);

	size_t operator()(const F_Vector& v1) const noexcept;

	void operator=(const F_Vector& copy);

	string ToString() const;

	float vx, vy; //Components
};

struct IndexPair {
	IndexPair() = default;
	IndexPair(unsigned int x_in, unsigned int y_in);

	bool operator==(const IndexPair& i1) const;

	unsigned int operator()(const IndexPair& i1) const noexcept;

	bool operator<(const IndexPair& i1) const;

	string ToString() const;

	unsigned int x = 0, y = 0; //Spots
};

class VectorField {
	public:
		/// <summary> 
		/// Default constructor. Has defaults for all dimension sizes as 1.
		/// </summary>
		VectorField(unsigned int x = 1, unsigned int y = 1);

		/// <summary> 
		/// Polymorphism constructor, used if there is a pre-loaded set argument.
		/// </summary>
		VectorField(unsigned int x, unsigned int y, const set<F_Vector>& set);

		/// <summary> 
		/// Gets a reference of the vector field map that contains the data.
		/// </summary>
		map<IndexPair, F_Vector>& GetVectorMap();

		/// <summary> 
		/// Operator overload for copying the data of an existing vector field.
		/// </summary>
		void operator=(const VectorField& copy);

		/// <summary> 
		/// Flattens the x direction of the map into a float pointer array.
		/// </summary>
		float* FlattenMapX();

		/// <summary> 
		/// Flattens the y direction of the map into a float pointer array.
		/// </summary>
		float* FlattenMapY();


		/// <summary> 
		/// Flattens the entire map into a CUDA provided float vector struct, where it is a pointer array.
		/// </summary>
		float3* FlattenMap();

		/// <summary> 
		/// Repacks/unflattens map using provided one-dimensional floater pointer arrays.
		/// </summary>
		void RepackMap(float* x, float* y);

		/// <summary> 
		/// Repacks/unflattens map using provided a float3 struct pointer array.
		/// </summary>
		void RepackMapVector(float3* vectors);

		/// <summary> 
		/// Returns an std::string of the corresponding keys (IndexPair) struct and the values (F_Vector) struct.
		/// </summary>
		string ToString() const;

	private:
		map<IndexPair, F_Vector> map_;
		unsigned int size_x_, size_y_;
		set<F_Vector> LoadDefaultVectorSet();
};