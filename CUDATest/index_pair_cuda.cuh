#pragma once

#include "cuda_runtime.h"

#include <string>

using std::string;

struct IndexPair {
	/// <summary> Default constructor, each dimension's default assignment is 0. </summary>
	__host__ __device__ IndexPair() = default;

	/// <summary> Loaded constructor, takes in an unsigned int for each dimension. </summary>
	__host__ __device__ IndexPair(unsigned int x_in, unsigned int y_in, unsigned int z_in, float weight_in = 0.33332f);

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

	/// <summary> Shifts the IndexPair incident frontwards one on the grid. </summary>
	__host__ __device__ IndexPair Front();

	/// <summary> Shifts the IndexPair incident backwards one on the grid. </summary>
	__host__ __device__ IndexPair Back();

	/// <summary> Shifts the IndexPair incident upwards one on the grid. </summary>
	__host__ __device__ IndexPair Up();

	/// <summary> Shifts the IndexPair incident downwards one on the grid. </summary>
	__host__ __device__ IndexPair Down();

	/// <summary> Shifts the IndexPair incident to the left, frontwards, and up one on the grid. </summary>
	__host__ __device__ IndexPair CornerLUpFront();

	/// <summary> Shifts the IndexPair incident to the left, frontwards, and down one on the grid. </summary>
	__host__ __device__ IndexPair CornerLDownFront();

	/// <summary> Shifts the IndexPair incident to the right, frontwards, and up one on the grid. </summary>
	__host__ __device__ IndexPair CornerRUpFront();

	/// <summary> Shifts the IndexPair incident to the right, frontwards, and down one on the grid. </summary>
	__host__ __device__ IndexPair CornerRDownFront();

	/// <summary> Shifts the IndexPair incident to the left, backwards, and up one on the grid. </summary>
	__host__ __device__ IndexPair CornerLUpBack();

	/// <summary> Shifts the IndexPair incident to the left, backwards, and down one on the grid. </summary>
	__host__ __device__ IndexPair CornerLDownBack();

	/// <summary> Shifts the IndexPair incident to the right, backwards, and up one on the grid. </summary>
	__host__ __device__ IndexPair CornerRUpBack();

	/// <summary> Shifts the IndexPair incident to the right, backwards, and down one on the grid. </summary>
	__host__ __device__ IndexPair CornerRDownBack();

	/// <summary> Shifts the IndexPair incident to the left and forward on the grid. </summary>
	__host__ __device__ IndexPair CornerLMidFront();

	/// <summary> Shifts the IndexPair incident to the right and forward on the grid. </summary>
	__host__ __device__ IndexPair CornerRMidFront();

	/// <summary> Shifts the IndexPair incident to the left and backward on the grid. </summary>
	__host__ __device__ IndexPair CornerLMidBack();

	/// <summary> Shifts the IndexPair incident to the right and backward on the grid. </summary>
	__host__ __device__ IndexPair CornerRMidBack();

	__host__ __device__ IndexPair MidUpFront();

	__host__ __device__ IndexPair MidUpBack();

	__host__ __device__ IndexPair MidUpLeft();

	__host__ __device__ IndexPair MidUpRight();

	__host__ __device__ IndexPair MidDownFront();

	__host__ __device__ IndexPair MidDownBack();

	__host__ __device__ IndexPair MidDownLeft();

	__host__ __device__ IndexPair MidDownRight();


	unsigned int x = 0, y = 0, z = 0; //Spots

	float weight;
};