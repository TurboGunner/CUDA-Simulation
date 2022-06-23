#pragma once

#include "cuda_runtime.h"

#include <string>

using std::string;

struct IndexPair {
	/// <summary> Default constructor, each dimension's default assignment is 0. </summary>
	__host__ __device__ IndexPair() = default;

	/// <summary> Loaded constructor, takes in an unsigned int for each dimension. </summary>
	__host__ __device__ IndexPair(unsigned int x_in, unsigned int y_in);

	/// <summary> Returns the effective linearized index of the grid. 
	/// <para> Made mostly for global/devices accesses due to CUDA being rather finicky with references. </para> </summary>
	__host__ __device__ size_t IX(size_t size) const;

	/// <summary> Operator overload for copying the data of an existing index pair. </summary>
	bool operator==(const IndexPair& i1) const;

	/// <summary> Operator overload for returning the proper hash code for IndexPair. </summary>
	size_t operator()(const IndexPair& i1) const noexcept;

	/// <summary>  Operator overload for comparing (less than). Required for proper functioning with maps. </summary>
	bool operator<(const IndexPair& i1) const;

	/// <summary> Returns an std::string of the coordinate values of IndexPair. </summary>
	string ToString() const;

	/// <summary> Shifts the IndexPair incident to the left on the grid. </summary>
	__host__ __device__ IndexPair Left();

	/// <summary> Shifts the IndexPair incident to the right on the grid. </summary>
	__host__ __device__ IndexPair Right();

	/// <summary> Shifts the IndexPair incident upwards one on the grid. </summary>
	__host__ __device__ IndexPair Up();

	/// <summary> Shifts the IndexPair incident downwards one on the grid. </summary>
	__host__ __device__ IndexPair Down();

	unsigned int x = 0, y = 0; //Spots
};