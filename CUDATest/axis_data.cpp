#include "axis_data.hpp"

#include <iostream>
#include <random>
#include <stdexcept>

inline int count = 0;

AxisData::AxisData(Axis axis) {
	axis_ = axis;
}

AxisData::AxisData(uint3 size, Axis axis) {
	if (size.x < 1 || size.y < 1 || size.z < 1) {
		throw std::invalid_argument("Error: Bounds must be at least greater than or equal to 1!");
	}

	size_ = size;
	axis_ = axis;
	map_ = new HashMap(size_.x * size_.y * size_.z);

	LoadDefaultDataSet();
}

void AxisData::LoadDefaultDataSet() {
	std::mt19937 generator(0);
	std::uniform_real_distribution<float> distribution(0.0f, 10.0f);
	unsigned int y_current = 0, z_current = 0;

	for (z_current; z_current < size_.z; z_current++) {
		for (y_current; y_current < size_.y; y_current++) {
			for (unsigned int i = 0; i < size_.x; i++) {
				float rand_float = distribution(generator);
				map_->Get(IndexPair(i, y_current, z_current).IX(size_.x)) = rand_float;
				total_ += rand_float;
			}
		}
	}
	std::cout << "AxisData Total: " << total_ << std::endl;
}

string AxisData::ToString() {
	string output;
	unsigned int y_current = 0, z_current = 0;
	for (z_current; z_current < size_.z; z_current++) {
		for (y_current; y_current < size_.y; y_current++) {
			for (unsigned int i = 0; i < size_.x; i++) {
				IndexPair current(i, y_current, z_current);
				output += current.ToString() + "\nValue: " + std::to_string(map_->Get(current)) + "\n\n";
			}
		}
	}
	return output;
}

void AxisData::operator=(const AxisData& copy) {
	map_ = copy.map_;
	axis_ = copy.axis_;
	size_ = copy.size_;
}

bool AxisData::operator<(const AxisData& copy) const {
	return !(axis_ == copy.axis_ && size_.x == copy.size_.x && size_.y == copy.size_.y && size_.z == copy.size_.z);
}