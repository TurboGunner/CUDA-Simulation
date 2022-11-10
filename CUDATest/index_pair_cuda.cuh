#pragma once

#include "cuda_runtime.h"

#include <string>

using std::string;

struct IndexPair {
	/// <summary> Default constructor, each dimension's default assignment is 0. </summary>
	__host__ __device__ IndexPair() = default;

	/// <summary> Loaded constructor, takes in an unsigned int for each dimension. </summary>
	__host__ __device__ IndexPair(unsigned int x_in, unsigned int y_in, unsigned int z_in, float weight_in = 0.33332f) { //Loaded Constructor
		x = x_in;
		y = y_in;
		z = z_in;

		weight = weight_in;
	}

	/// <summary> Returns the effective linearized index of the grid. 
	/// <para> Made mostly for global/device accesses due to CUDA being rather finicky with references. </para> </summary>
	__host__ __device__ size_t IX(size_t size) const {
		return x + (y * size) + (z * (size * size));
	}

	/// <summary> Returns the effective linearized index of the grid, using a non-square boundary. 
	/// <para> Made mostly for global/devices accesses due to CUDA being rather finicky with references. </para> </summary>
	__host__ __device__ size_t IX(uint3 size) const {
		return x + (y * size.x) + (z * (size.y * size.z));
	}

	/// <summary> Operator overload for copying the data of an existing index pair. </summary>
	bool operator==(const IndexPair& i1) const {
		return x == i1.x && y == i1.y;
	}

	/// <summary> Operator overload for returning the proper hash code for IndexPair. </summary>
	__host__ size_t operator()(const IndexPair& i1) const noexcept {
		unsigned int hash1 = std::hash<unsigned int>()(i1.x);
		unsigned int hash2 = std::hash<unsigned int>()(i1.y);
		return hash1 ^ (hash2 << 1);
	}

	/// <summary>  Operator overload for comparing (less than). Required for proper functioning with maps. </summary>
	bool operator<(const IndexPair& i1) const {
		if (y == i1.y) {
			return x < i1.x;
		}
		return y < i1.y;
	}

	/// <summary> Returns an std::string of the coordinate values of IndexPair. </summary>
	string ToString() const {
		return "X Component: " + std::to_string(x) + " | Y Component: " + std::to_string(y);
	}

	/// <summary> Shifts the IndexPair incident to the left on the grid. </summary>
	__host__ __device__ IndexPair Left() {
		return IndexPair(x - 1, y, z, 0.05556f);
	}


	/// <summary> Shifts the IndexPair incident to the right on the grid. </summary>
	__host__ __device__ IndexPair Right() {
		return IndexPair(x + 1, y, z, 0.05556f);
	}

	/// <summary> Shifts the IndexPair incident frontwards one on the grid. </summary>
	__host__ __device__ IndexPair Front() {
		return IndexPair(x, y + 1, z, 0.05556f);
	}

	/// <summary> Shifts the IndexPair incident backwards one on the grid. </summary>
	__host__ __device__ IndexPair Back() {
		return IndexPair(x, y - 1, z, 0.05556f);
	}

	/// <summary> Shifts the IndexPair incident upwards one on the grid. </summary>
	__host__ __device__ IndexPair Up() {
		return IndexPair(x, y, z + 1, 0.05556f);
	}

	/// <summary> Shifts the IndexPair incident downwards one on the grid. </summary>
	__host__ __device__ IndexPair Down() {
		return IndexPair(x, y, z - 1, 0.05556f);
	}

	/// <summary> Shifts the IndexPair incident to the left, frontwards, and up one on the grid. </summary>
	__host__ __device__ IndexPair CornerLUpFront() {
		return IndexPair(x - 1, y + 1, z + 1, 0.027778f);
	}

	/// <summary> Shifts the IndexPair incident to the left, frontwards, and down one on the grid. </summary>
	__host__ __device__ IndexPair CornerLDownFront() {
		return IndexPair(x - 1, y + 1, z - 1, 0.027778f);
	}

	/// <summary> Shifts the IndexPair incident to the right, frontwards, and up one on the grid. </summary>
	__host__ __device__ IndexPair CornerRUpFront() {
		return IndexPair(x + 1, y + 1, z + 1, 0.027778f);
	}

	/// <summary> Shifts the IndexPair incident to the right, frontwards, and down one on the grid. </summary>
	__host__ __device__ IndexPair CornerRDownFront() {
		return IndexPair(x + 1, y + 1, z - 1, 0.027778f);
	}

	/// <summary> Shifts the IndexPair incident to the left, backwards, and up one on the grid. </summary>
	__host__ __device__ IndexPair CornerLUpBack() {
		return IndexPair(x - 1, y - 1, z + 1, 0.027778f);
	}

	/// <summary> Shifts the IndexPair incident to the left, backwards, and down one on the grid. </summary>
	__host__ __device__ IndexPair CornerLDownBack() {
		return IndexPair(x - 1, y - 1, z - 1, 0.027778f);
	}


	/// <summary> Shifts the IndexPair incident to the right, backwards, and up one on the grid. </summary>
	__host__ __device__ IndexPair CornerRUpBack() {
		return IndexPair(x + 1, y - 1, z + 1, 0.027778f);
	}

	/// <summary> Shifts the IndexPair incident to the right, backwards, and down one on the grid. </summary>
	__host__ __device__ IndexPair CornerRDownBack() {
		return IndexPair(x + 1, y - 1, z - 1, 0.027778f);
	}

	/// <summary> Shifts the IndexPair incident to the left and forward on the grid. </summary>
	__host__ __device__ IndexPair CornerLMidFront() {
		return IndexPair(x - 1, y + 1, z, 0.027778f);
	}


	/// <summary> Shifts the IndexPair incident to the right and forward on the grid. </summary>
	__host__ __device__ IndexPair CornerRMidFront() {
		return IndexPair(x + 1, y + 1, z, 0.027778f);
	}


	/// <summary> Shifts the IndexPair incident to the left and backward on the grid. </summary>
	__host__ __device__ IndexPair CornerLMidBack() {
		return IndexPair(x - 1, y - 1, z, 0.027778f);
	}


	/// <summary> Shifts the IndexPair incident to the right and backward on the grid. </summary>

	__host__ __device__ IndexPair CornerRMidBack() {
		return IndexPair(x + 1, y - 1, z, 0.027778f);
	}


	__host__ __device__ IndexPair MidUpFront() {
		return IndexPair(x, y - 1, z + 1);
	}

	__host__ __device__ IndexPair MidDownFront() {
		return IndexPair(x, y - 1, z - 1);
	}

	//Mid Back Indexes

	__host__ __device__ IndexPair MidUpBack() {
		return IndexPair(x, y - 1, z + 1);
	}

	__host__ __device__ IndexPair MidDownBack() {
		return IndexPair(x, y - 1, z - 1);
	}

	//Mid Left Indexes

	__host__ __device__ IndexPair MidUpLeft() {
		return IndexPair(x - 1, y, z + 1);
	}

	__host__ __device__ IndexPair MidDownLeft() {
		return IndexPair(x - 1, y, z - 1);
	}

	//Mid Right Indexes

	__host__ __device__ IndexPair MidUpRight() {
		return IndexPair(x + 1, y, z + 1);
	}

	__host__ __device__ IndexPair MidDownRight() {
		return IndexPair(x + 1, y, z - 1);
	}


	unsigned int x = 0, y = 0, z = 0; //Spots

	float weight;
};