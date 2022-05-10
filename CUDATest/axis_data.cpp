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
	size_ = size;
	axis_ = axis;
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
	for (y_current; y_current < size_; y_current++) {
		for (unsigned int i = 0; i < size_; i++) {
			map_.emplace(IndexPair(i, y_current), RandomFloat());
		}
	}
}

float* AxisData::FlattenMap() {
	float* arr = new float[map_.size()];

	unsigned int count = 0;
	unsigned int size = (unsigned int)sqrt(map_.size());
	unsigned int y_current = 0;

	for (y_current; y_current < size; y_current++) {
		for (unsigned int i = 0; i < size; i++) {
			IndexPair current(i, y_current);
			arr[count] = map_[current];
			count++;
		}
	}
	return arr;
}

string AxisData::ToString() {
	string output;
	unsigned int size = (unsigned int)sqrt(map_.size());
	unsigned int y_current = 0;
	for (y_current; y_current < size; y_current++) {
		for (unsigned int i = 0; i < size; i++) {
			std::cout << i << std::endl;
			IndexPair current(i, y_current);
			output += current.ToString() + "\nValue: " + std::to_string(map_[current]) + "\n\n";
		}
	}
	return output;
}

void AxisData::operator=(const AxisData& copy) {
	map_ = copy.map_;
	axis_ = copy.axis_;
	size_ = copy.size_;
}

void AxisData::RepackMap(float* data) {
	unsigned int y_current = 0;
	unsigned int size = (unsigned int)sqrt(map_.size());
	unsigned int count = 0;

	for (y_current; y_current < size; y_current++) {
		for (unsigned int i = 0; i < size; i++) {
			IndexPair current(i, y_current);
			map_[IndexPair(i, y_current)] = data[count];
			count++;
		}
	}
}