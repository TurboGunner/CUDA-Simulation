#include "vector_field.hpp"

#include <stdexcept>
#include <iostream>

using std::string;
using std::set;
using std::map;

//Constructors

VectorField::VectorField(unsigned int x, unsigned int y) {
	if (x == 0 || y == 0) {
		throw std::invalid_argument("Error: Bounds must be at least greater than or equal to 1!");
	}
	size_x_ = x;
	size_y_ = y;
	LoadDefaultVectorSet();
}

VectorField::VectorField(unsigned int x, unsigned int y, const set<F_Vector>& set) {
	if (x == 0 || y == 0) {
		throw std::invalid_argument("Error: Bounds must be at least greater than or equal to 1!");
	}
	size_x_ = x;
	size_y_ = y;
	LoadDefaultVectorSet();
	unsigned int xCurrent = 0, yCurrent = 0;
	for (auto const& element : set) {
		if (xCurrent == size_x_) {
			xCurrent = 0;
			yCurrent++;
		}
		IndexPair pair(xCurrent, yCurrent);
		map_.at(pair) = element;
		xCurrent++;
	}
}

set<F_Vector> VectorField::LoadDefaultVectorSet() {
	set<F_Vector> output;
	unsigned int y_current = 0;
	for (F_Vector vector(1, 1); y_current < size_y_; y_current++) {
		for (unsigned int i = 0; i < size_x_; i++) {
			IndexPair pair(i, y_current);
			map_.emplace(pair, vector);
		}
	}
	return output;
}

string VectorField::ToString() {
	string output;
	for (auto const& entry : map_) {
		output += entry.first.ToString() + "\n" + entry.second.ToString() + "\n\n";
	}
	return output;
}

unsigned int VectorField::GetSizeX() const {
	return size_x_;
}

unsigned int VectorField::GetSizeY() const {
	return size_y_;
}

void VectorField::operator=(const VectorField& copy) {
	map_ = copy.map_;
}

map<IndexPair, F_Vector>& VectorField::GetVectorMap() {
	return map_;
}

float* VectorField::FlattenMapX() {
	float* arr = new float[map_.size()];
	unsigned int count = 0;
	for (const auto& entry : map_) {
		arr[count] = entry.second.vx;
		count++;
	}
	return arr;
}

float* VectorField::FlattenMapY() {
	float* arr = new float[size_x_ * size_y_];
	unsigned int y_current = 0, count = 0;

	for (y_current; y_current < size_y_; y_current++) {
		for (unsigned int i = 0; i < size_x_; i++) {
			arr[i * (y_current + 1)] = map_[IndexPair(i, y_current)].vy;
		}
	}
	return arr;
}