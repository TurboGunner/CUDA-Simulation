#pragma once

#include "cuda_runtime.h"

#include "index_pair_cuda.cuh"
#include "axis_data.hpp"
#include "cudamap.cuh"

#include <string>

using std::string;

class VectorField {
	public:
		/// <summary> Default constructor. </summary>
		VectorField() = default;

		/// <summary> Loaded constructor for the size initialization of the vector field. </summary>
		VectorField(uint3 size);

		/// <summary> Gets a reference of the vector field map that contains the data. </summary>
		AxisData*& GetVectorMap();

		/// <summary> Operator overload for copying the data of an existing vector field. </summary>
		void operator=(const VectorField& copy);

		/// <summary> Returns an std::string of the corresponding keys (IndexPair) struct and the values (F_Vector) struct. </summary>
		string ToString();

	private:
		AxisData* map_;
		uint3 size_;

		/// <summary> Loads in zeroes for all data. Made to be sparse. </summary>
		void LoadDefaultVectorSet();
};