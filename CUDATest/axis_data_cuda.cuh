#pragma once

#include "index_pair.hpp"

#include <unordered_map>

struct HashFunction {
	size_t operator()(const IndexPair& i1) const {
		unsigned int hash1 = std::hash<unsigned int>()(i1.x);
		unsigned int hash2 = std::hash<unsigned int>()(i1.y);
		return hash1 ^ (hash2 << 1);
	}
};

struct KeyHash {
	bool operator()(const IndexPair& i1, const IndexPair& i2) const {
		return i1 == i2;
	}
};

enum class Axis { X, Y, Z };

struct AxisDataGPU {
	/// <summary> 
	/// Default constructor, zero initialized parameters.
	/// </summary>
	AxisDataGPU() = default;

	/// <summary> 
	/// Constructor, contains the axis, used for density (or general axis-dependent data).
	/// </summary>
	AxisDataGPU(Axis axis);

	/// <summary> 
	/// Main loaded constructor, contains both size initialization and axis initialization (for creation of new axis-dependent data elements).
	/// <para> An example of this is in the initialiation of the FluidSim object, where it will create new density AxisData objects based on the input size. </para>
	/// </summary>
	AxisDataGPU(unsigned int size, Axis axis = Axis::X); //For density

	// <summary> 
	/// Main loaded constructor, contains both size initialization and axis initialization (for creation of new axis-dependent data elements).
	/// <para> An example of this is in the initialiation of the FluidSim object, where it will create new density AxisData objects based on the input size. </para>
	/// </summary>
	void LoadDefaultDataSet();

	// <summary> 
	/// Copy constructor for AxisDataGPU.
	/// </summary>
	void operator=(const AxisDataGPU& copy);

	Axis axis_;
	std::unordered_map<IndexPair, float, HashFunction, KeyHash> map_;
	unsigned int size_;

};