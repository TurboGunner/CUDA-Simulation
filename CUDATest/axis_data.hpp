#pragma once

#include "index_pair_cuda.cuh"

#include "cudamap.cuh"

enum class Axis { X, Y, Z };

struct AxisData {
	/// <summary> Default constructor. </summary>
	AxisData() = default;

	/// <summary> Loaded constructor, has no size allocation, meant to be initialized later for the size. </summary>
	AxisData(Axis axis);

	/// <summary> Loaded constructor, has size allocation parameters, and a default parameter for the axis. </summary>
	AxisData(unsigned int size_x, unsigned int size_y, Axis axis = Axis::X); //For density

	/// <summary> Outputs float values bundled together into one std::string output. </summary>
	string ToString();

	/// <summary> Operator overload for the copy function. </summary>
	void operator=(const AxisData& copy);

	Axis axis_;
	HashMap<float>* map_;

	private:
		unsigned int size_x_, size_y_;
		/// <summary> Loads in zeroes for all data. Made to be sparse. </summary>
		void LoadDefaultDataSet();
};