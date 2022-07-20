#pragma once

#include "cuda_runtime.h"

#include <string>

using std::string;

struct IndexPair {
	/// <summary> Default constructor, each dimension's default assignment is 0. </summary>
	__host__ __device__ IndexPair() = default;

	/// <summary> Loaded constructor, takes in an unsigned int for each dimension. </summary>
	__host__ __device__ IndexPair(unsigned int x_in, unsigned int y_in, unsigned int z_in, float weight_in = (1/3));

	/// <summary> Returns the effective linearized index of the grid. 
	/// <para> Made mostly for global/device accesses due to CUDA being rather finicky with references. </para> </summary>
	__host__ __device__ size_t IX(size_t size) const;

	/// <summary> Returns the effective linearized index of the grid, using a non-square boundary. 
	/// <para> Made mostly for global/devices accesses due to CUDA being rather finicky with references. </para> </summary>
	__host__ __device__ size_t IX(uint3 size) const;

	/// <summary> Operator overload for copying the data of an existing index pair. </summary>
	bool operator==(const IndexPair& i1) const;

	/// <summary> Operator overload for returning the proper hash code for IndexPair. </summary>
	__host__ size_t operator()(const IndexPair& i1) const noexcept;

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

	/// <summary> Shifts the IndexPair incident frontwards one on the grid. </summary>
	__host__ __device__ IndexPair Front();

	/// <summary> Shifts the IndexPair incident backwards one on the grid. </summary>
	__host__ __device__ IndexPair Back();

	__host__ __device__ IndexPair CornerLUpFront();

	__host__ __device__ IndexPair CornerLDownFront();

	__host__ __device__ IndexPair CornerRUpFront();

	__host__ __device__ IndexPair CornerRDownFront();

	__host__ __device__ IndexPair CornerLUpBack();

	__host__ __device__ IndexPair CornerLDownBack();

	__host__ __device__ IndexPair CornerRUpBack();

	__host__ __device__ IndexPair CornerRDownBack();

	__host__ __device__ IndexPair CornerLMidFront();

	__host__ __device__ IndexPair CornerRMidFront();

	__host__ __device__ IndexPair CornerLMidBack();

	__host__ __device__ IndexPair CornerRMidBack();

	unsigned int x = 0, y = 0, z = 0; //Spots

	float weight = 1 / 3;

	const float first_n_weight = 0.05556f,
		second_n_weight = 0.027778f;
};