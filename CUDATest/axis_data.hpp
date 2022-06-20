#pragma once

#include "index_pair_cuda.cuh"

#include "cudamap.cuh"

enum class Axis { X, Y, Z };

struct AxisData {
	AxisData() = default;
	AxisData(Axis axis);
	AxisData(unsigned int size, Axis axis = Axis::X); //For density

	void LoadDefaultDataSet();

	string ToString();

	void operator=(const AxisData& copy);

	Axis axis_;
	HashMap<float>* map_;

	unsigned int size_;
};