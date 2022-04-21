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
	float* arr = new float[map_.size()];
	unsigned int count = 0;
	for (const auto& entry : map_) {
		arr[count] = entry.second.vy;
		count++;
	}
	return arr;
}

void VectorField::RepackMap(float* x, float* y) {
	unsigned int count = 0;
	for (auto& entry : map_) {
		entry.second = F_Vector(x[count], y[count]);
		count++;
	}
}