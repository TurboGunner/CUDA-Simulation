#pragma once

#include "cuda_runtime.h"

#include "f_vector.hpp"
#include "index_pair_cuda.cuh"
#include "axis_data.hpp"
#include "cudamap.cuh"

#include <string>
#include <set>

using std::string;
using std::set;

template <typename K>
struct Hash {
	__host__ __device__ size_t operator()(const K& i1, size_t size) const {
		return (((IndexPair)i1).x ) + (((IndexPair)i1).y * (sqrt(size)));
	}
};

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
		HashMap<IndexPair, F_Vector, Hash<IndexPair>>*& GetVectorMap();

		/// <summary> 
		/// Operator overload for copying the data of an existing vector field.
		/// </summary>
		void operator=(const VectorField& copy);

		/// <summary> 
		/// Returns an std::string of the corresponding keys (IndexPair) struct and the values (F_Vector) struct.
		/// </summary>
		string ToString();

		/// <summary> 
		/// Returns an AxisData struct that contains all data points from a given axis.
		/// </summary>
		void DataConstrained(Axis axis, AxisData& input);

		/// <summary> 
		/// Repacks the AxisData struct with the current VectorField instance.
		/// </summary>
		void RepackFromConstrained(AxisData& axis);

	private:
		HashMap<IndexPair, F_Vector, Hash<IndexPair>>* map_;
		unsigned int size_x_, size_y_;
		set<F_Vector> LoadDefaultVectorSet();
};