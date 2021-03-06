#include "vector_field.hpp"

#include <stdexcept>
#include <iostream>

//Constructors

VectorField::VectorField(uint3 size) {
	if (size.x == 0 || size.y == 0 || size.z == 0) {
		throw std::invalid_argument("Error: Bounds must be at least greater than or equal to 1!");
	}

	size_ = size;
	AxisData axis_x(size_, Axis::X), axis_y(size_, Axis::Y), axis_z(size_, Axis::Z);

	std::cout << axis_x.total_ << std::endl;

	map_.insert(map_.begin(), { axis_x, axis_y, axis_z });
}

string VectorField::ToString() {
	string output;
	unsigned int y_current = 0, z_current = 0;

	for (z_current; z_current < size_.z; z_current++) {
		for (y_current; y_current < size_.y; y_current++) {
			for (unsigned int i = 0; i < size_.x; i++) {
				IndexPair current(i, y_current, z_current);
				output += current.ToString() + "\n" +
					std::to_string(map_[0].map_->Get(current.IX(size_.x)))
					+ " | "
					+ std::to_string(map_[1].map_->Get(current.IX(size_.y)))
					+ " | "
					+ std::to_string(map_[2].map_->Get(current.IX(size_.z)))
					+ "\n\n";
			}
		}
	}
	return output;
}

void VectorField::operator=(const VectorField& copy) {
	if (copy.map_ == this->map_) {
		return;
	}
	map_ = copy.map_;

	size_ = copy.size_;
}