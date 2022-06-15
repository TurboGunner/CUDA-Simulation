#pragma once

#include "index_pair.hpp"

#include "cudamap.cuh"

template <typename K>
struct HashDupe {
	__host__ __device__ size_t operator()(const K& i1, size_t size) const {
		return ((IndexPair)i1).x ^ (((IndexPair)i1).y << 1) % size;
	}
};

enum class Axis { X, Y, Z };

struct AxisData {
	AxisData() = default;
	AxisData(Axis axis);
	AxisData(unsigned int size, Axis axis = Axis::X); //For density

	void LoadDefaultDataSet();

	string ToString();

	void operator=(const AxisData& copy);

	Axis axis_;
	HashMap<IndexPair, float, HashDupe<IndexPair>>* map_;

	unsigned int size_;
};