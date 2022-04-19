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
	unsigned int yCurrent = 0;
	for (F_Vector vector(1, 1); yCurrent < size_y_; yCurrent++) {
		for (unsigned int i = 0; i < size_x_; i++) {
			IndexPair pair(i, yCurrent);
			map_.emplace(pair, vector);
		}
	}
	return output;
}

string VectorField::ToString() {
	string output;
	std::cout << map_.size() << std::endl;
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