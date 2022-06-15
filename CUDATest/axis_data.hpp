#pragma once

#include "index_pair.hpp"
#include "cudamap.cuh"

enum class Axis { X, Y, Z };

struct HashDupe {
	__host__ __device__ size_t operator()(const IndexPair& i1) const {
		return i1.x ^ (i1.y << 1);
	}
};

struct AxisData {
	AxisData() = default;
	AxisData(Axis axis);
	AxisData(unsigned int size, Axis axis = Axis::X); //For density

	void LoadDefaultDataSet();

	string ToString();

	void operator=(const AxisData& copy);

	Axis axis_;
	HashMap<IndexPair, float, HashDupe>* map_;

	unsigned int size_;
};