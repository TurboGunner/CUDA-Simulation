#include "axis_data.hpp"

#include <iostream>
#include <stdexcept>

AxisData::AxisData(Axis axis) {
	axis_ = axis;
	LoadDefaultDataSet();
}

AxisData::AxisData(unsigned int size_x, unsigned int size_y, Axis axis) {
	if (size_x < 1 || size_y < 1) {
		throw std::invalid_argument("Error: Bounds must be at least greater than or equal to 1!");
	}
	size_x_ = size_x;
	size_y_ = size_y;
	axis_ = axis;
	map_ = new HashMap<float>(size_x_ * size_y_);
	LoadDefaultDataSet();
}

void AxisData::LoadDefaultDataSet() {
	unsigned int y_current = 0;
	for (y_current; y_current < size_y_; y_current++) {
		for (unsigned int i = 0; i < size_x_; i++) {
			map_->Put(IndexPair(i, y_current), 0);
		}
	}
}

string AxisData::ToString() {
	string output;
	unsigned int y_current = 0;
	for (y_current; y_current < size_y_; y_current++) {
		for (unsigned int i = 0; i < size_x_; i++) {
			IndexPair current(i, y_current);
			output += current.ToString() + "\nValue: " + std::to_string(map_->Get(current)) + "\n\n";
		}
	}
	return output;
}

void AxisData::operator=(const AxisData& copy) {
	map_ = copy.map_;
	axis_ = copy.axis_;
	size_x_ = copy.size_x_;
	size_y_ = copy.size_y_;
}