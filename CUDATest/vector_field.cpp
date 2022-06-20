#include "vector_field.hpp"

#include <stdexcept>
#include <iostream>

//Constructors

VectorField::VectorField(unsigned int x, unsigned int y) {
	if (x == 0 || y == 0) {
		throw std::invalid_argument("Error: Bounds must be at least greater than or equal to 1!");
	}
	size_x_ = x;
	size_y_ = y;
	map_ = new HashMap<F_Vector>(size_x_ * size_y_);
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
		map_->Put(pair, element);
		xCurrent++;
	}
}

set<F_Vector> VectorField::LoadDefaultVectorSet() {
	set<F_Vector> output;
	unsigned int y_current = 0;
	for (y_current; y_current < size_y_; y_current++) {
		for (unsigned int i = 0; i < size_x_; i++) {
			IndexPair pair(i, y_current);
			map_->Put(pair, RandomVector());
			//std::cout << (y_current * size_y_) + i << std::endl;
		}
	}
	return output;
}

string VectorField::ToString() {
	string output;
	unsigned int size = size_x_;
	unsigned int y_current = 0;
	for (y_current; y_current < size; y_current++) {
		for (unsigned int i = 0; i < size; i++) {
			IndexPair current(i, y_current);
			output += current.ToString() + "\n" + map_->Get(current.IX(size)).ToString() + "\n\n";
		}
	}
	return output;
}

void VectorField::operator=(const VectorField& copy) {
	map_ = map_ = new HashMap<F_Vector>(copy.size_x_ * copy.size_y_);
	map_ = copy.map_;

	size_x_ = copy.size_x_;
	size_y_ = copy.size_y_;
}

HashMap<F_Vector>*& VectorField::GetVectorMap() {
	return map_;
}

void VectorField::DataConstrained(Axis axis, AxisData& input) {
	unsigned int size = size_x_;
	unsigned int y_current = 0;
	for (y_current; y_current < size; y_current++) {
		for (unsigned int i = 0; i < size; i++) {
			IndexPair current(i, y_current);
			float current_float;
			if (axis == Axis::X) {
				current_float = map_->Get(current.IX(size)).vx_;
			}
			else {
				current_float = map_->Get(current.IX(size)).vy_;
			}
			input.map_->Put(current.IX(size), current_float);
		}
	}
}

void VectorField::RepackFromConstrained(AxisData& axis) {
	unsigned int size = size_x_;
	unsigned int y_current = 0;
	for (y_current; y_current < size; y_current++) {
		for (unsigned int i = 0; i < size; i++) {
			IndexPair current(i, y_current);
			if (axis.axis_ == Axis::X) {
				map_->Get(current.IX(size)).vx_ = axis.map_->Get(current.IX(size));
			}
			else {
				map_->Get(current.IX(size)).vy_ = axis.map_->Get(current.IX(size));
			}
			//std::cout << current.ToString() << std::endl;
		}
	}
	//std::cout << "Last element!" << std::endl;
}