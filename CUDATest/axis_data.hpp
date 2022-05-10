#pragma once

#include "index_pair.hpp"

#include <unordered_map>
using std::unordered_map;

enum class Axis { X, Y, Z };

struct HashDupe {
	size_t operator()(const IndexPair& i1) const {
		unsigned int hash1 = std::hash<unsigned int>()(i1.x);
		unsigned int hash2 = std::hash<unsigned int>()(i1.y);
		return hash1 ^ (hash2 << 1);
	}
};

struct AxisData {
	AxisData() = default;
	AxisData(Axis axis);
	AxisData(unsigned int size, Axis axis = Axis::X); //For density

	void LoadDefaultDataSet();

	float* FlattenMap();
	void RepackMap(float* data);
	string ToString();

	void operator=(const AxisData& copy);

	Axis axis_;
	unordered_map<IndexPair, float, HashDupe> map_;

	unsigned int size_;
};