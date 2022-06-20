#include "axis_data.hpp"

#include <iostream>
#include <random>
#include <stdexcept>

AxisData::AxisData(Axis axis) {
	axis_ = axis;
	LoadDefaultDataSet();
}

AxisData::AxisData(unsigned int size, Axis axis) {
	if (size < 1) {
		throw std::invalid_argument("Error: Bounds must be at least greater than or equal to 1!");
	}
	size_ = size * size;
	axis_ = axis;
	map_ = new HashMap<float>(size_);
	LoadDefaultDataSet();
}

inline float RandomFloat() {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(1, 10);
	return dist(gen);
}

void AxisData::LoadDefaultDataSet() {
	unsigned int y_current = 0;
	for (y_current; y_current < sqrt(size_); y_current++) {
		for (unsigned int i = 0; i < sqrt(size_); i++) {
			map_->Put(IndexPair(i, y_current), RandomFloat());
		}
	}
}

string AxisData::ToString() {
	string output;
	unsigned int y_current = 0;
	for (y_current; y_current < sqrt(size_); y_current++) {
		for (unsigned int i = 0; i < sqrt(size_); i++) {
			IndexPair current(i, y_current);
			output += current.ToString() + "\nValue: " + std::to_string(map_->Get(current)) + "\n\n";
		}
	}
	return output;
}

void AxisData::operator=(const AxisData& copy) {
	map_ = copy.map_;
	axis_ = copy.axis_;
	size_ = copy.size_;
}