#pragma once

#include "index_pair_cuda.cuh"

#include "cudamap.cuh"

#include "handler_classes.hpp"

enum class Axis { X, Y, Z };

static RandomFloat random(0.0f, 10.0f);

/// <summary> Acts as either a container for scalar quantities (density), or as a constituent axis for a vector quantity (velocity). </summary>
struct AxisData {
	AxisData() = default;
	/// <summary> Loaded constructor, has no size allocation, meant to be initialized later for the size. </summary>
	AxisData(Axis axis);

	/// <summary> Loaded constructor, has size allocation parameters, and a default parameter for the axis. </summary>
	AxisData(uint3 size, Axis axis = Axis::X); //For density

	/// <summary> Outputs float values bundled together into one std::string output. </summary>
	string ToString();

	/// <summary> Operator overload for the copy function. </summary>
	void operator=(const AxisData& copy);

	bool operator==(const AxisData& copy) const noexcept {
		return axis_ == copy.axis_ && size_.x == copy.size_.x && size_.y == copy.size_.y && size_.z == copy.size_.z;
	}

	friend bool operator==(AxisData& a1, AxisData& a2) {
		return a1.axis_ == a2.axis_ && a1.size_.x == a2.size_.x && a1.size_.y == a2.size_.y && a1.size_.z == a2.size_.z;
	}

	bool operator<(const AxisData& copy) const;

	Axis axis_;
	HashMap* map_;

	float total_ = 0.0f;

private:
	/// <summary> Loads in zeroes for all data. Made to be sparse. </summary>
	void LoadDefaultDataSet();

	uint3 size_;
};