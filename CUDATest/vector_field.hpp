#pragma once

#include "cuda_runtime.h"

#include "f_vector.hpp"
#include "index_pair.hpp"

#include <string>
#include <set>
#include <map>

using std::string;
using std::set;
using std::map;

class VectorField {
	public:
		/// <summary> 
		/// Default constructor. Has defaults for all dimension sizes as 1.
		/// </summary>
		VectorField(unsigned int x = 1, unsigned int y = 1);

		/// <summary> 
		/// Loaded constructor, used if there is a pre-loaded set argument.
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
		string ToString();

	private:
		map<IndexPair, F_Vector> map_;
		unsigned int size_x_, size_y_;
		set<F_Vector> LoadDefaultVectorSet();
};