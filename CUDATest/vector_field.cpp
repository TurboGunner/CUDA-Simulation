#include "vector_field.hpp"

#include <stdexcept>
#include <iostream>
#include <random>

//Constructors

inline float RandomFloat() {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(1, 10);
	return dist(gen);
}

VectorField::VectorField(unsigned int x, unsigned int y) {
	if (x == 0 || y == 0) {
		throw std::invalid_argument("Error: Bounds must be at least greater than or equal to 1!");
	}
	size_x_ = x;
	size_y_ = y;
	map_ = new AxisData[2];
	AxisData axis_x(size_x_, size_y_, Axis::X), axis_y(size_x_, size_y_, Axis::Y);
	map_[0] = axis_x;
	map_[1] = axis_y;
	LoadDefaultVectorSet();
}

void VectorField::LoadDefaultVectorSet() {
	unsigned int y_current = 0;
	for (y_current; y_current < size_y_; y_current++) {
		for (unsigned int i = 0; i < size_x_; i++) {
			IndexPair pair(i, y_current);
			map_[0].map_->Put(pair, RandomFloat());
			map_[1].map_->Put(pair, RandomFloat());
		}
	}
}

string VectorField::ToString() {
	string output;
	unsigned int y_current = 0;
	for (y_current; y_current < size_y_; y_current++) {
		for (unsigned int i = 0; i < size_x_; i++) {
			IndexPair current(i, y_current);
			output += current.ToString() + "\n" + 
				std::to_string(map_[0].map_->Get(current.IX(size))) 
				+ " | "
				+ std::to_string(map_[1].map_->Get(current.IX(size))) 
				+ "\n\n";
		}
	}
	return output;
}

void VectorField::operator=(const VectorField& copy) {
	if (copy.map_ == this->map_) {
		return;
	}
	map_ = copy.map_;

	size_x_ = copy.size_x_;
	size_y_ = copy.size_y_;
}

AxisData*& VectorField::GetVectorMap() {
	return map_;
}