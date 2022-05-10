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
	for (y_current; y_current < size_y_; y_current++) {
		for (unsigned int i = 0; i < size_x_; i++) {
			IndexPair pair(i, y_current);
			map_.emplace(pair, RandomVector());
		}
	}
	return output;
}

string VectorField::ToString() {
	string output;
	unsigned int size = (unsigned int) sqrt(map_.size());
	unsigned int y_current = 0;
	for (y_current; y_current < size; y_current++) {
		for (unsigned int i = 0; i < size; i++) {
			IndexPair current(i, y_current);
			output += current.ToString() + "\n" + map_[current].ToString() + "\n\n";
		}
	}
	return output;
}

void VectorField::operator=(const VectorField& copy) {
	map_ = copy.map_;

	size_x_ = copy.size_x_;
	size_y_ = copy.size_y_;
}

unordered_map<IndexPair, F_Vector, Hash>& VectorField::GetVectorMap() {
	return map_;
}

float* VectorField::FlattenMapX() {
	float* arr = new float[map_.size()];

	unsigned int count = 0;
	unsigned int size = (unsigned int)sqrt(map_.size());
	unsigned int y_current = 0;

	for (y_current; y_current < size; y_current++) {
		for (unsigned int i = 0; i < size; i++) {
			IndexPair current(i, y_current);
			arr[count] = map_[current].vx;
			count++;
		}
	}
	return arr;
}

float* VectorField::FlattenMapY() {
	float* arr = new float[map_.size()];

	unsigned int count = 0;
	unsigned int size = (unsigned int) sqrt(map_.size());
	unsigned int y_current = 0;

	for (y_current; y_current < size; y_current++) {
		for (unsigned int i = 0; i < size; i++) {
			IndexPair current(i, y_current);
			arr[count] = map_[current].vy;
			count++;
		}
	}
	return arr;
}

float3* VectorField::FlattenMap() {
	unsigned int bound = map_.size();
	float* x_dim = FlattenMapX(),
		*y_dim = FlattenMapY();
	float3* vectors = new float3[bound];
	for (int i = 0; i < bound; i++) {
		vectors[i] = float3(x_dim[i], y_dim[i]);
	}
	free(x_dim);
	free(y_dim);

	return vectors;
}

void VectorField::RepackMap(float* x, float* y) {
	unsigned int y_current = 0;
	unsigned int size = (unsigned int)sqrt(map_.size());
	unsigned int count = 0;

	for (y_current; y_current < size; y_current++) {
		for (unsigned int i = 0; i < size; i++) {
			IndexPair current(i, y_current);
			map_[IndexPair(i, y_current)] = F_Vector(x[count], y[count]);
			std::cout << y[count] << std::endl;
			count++;
		}
	}
}

void VectorField::RepackMapVector(float3* vectors) {
	unsigned int y_current = 0;
	unsigned int size = (unsigned int)sqrt(map_.size());
	unsigned int count = 0;

	for (y_current; y_current < size; y_current++) {
		for (unsigned int i = 0; i < size; i++) {
			IndexPair current(i, y_current);
			map_[IndexPair(i, y_current)] = F_Vector(vectors[count].x, vectors[count].y);
			count++;
		}
	}
}

void VectorField::DataConstrained(Axis axis, AxisData& input) {
	unsigned int size = (unsigned int)sqrt(map_.size());
	unsigned int y_current = 0;
	for (y_current; y_current < size; y_current++) {
		for (unsigned int i = 0; i < size; i++) {
			IndexPair current(i, y_current);
			float current_float;
			if (axis == Axis::X) {
				current_float = map_[current].vx;
			}
			else {
				current_float = map_[current].vy;
			}
			input.map_.emplace(current, current_float);
		}
	}
}

void VectorField::RepackFromConstrained(AxisData& axis) {
	unsigned int size = (unsigned int)sqrt(map_.size());
	unsigned int y_current = 0;
	for (y_current; y_current < size; y_current++) {
		for (unsigned int i = 0; i < size; i++) {
			IndexPair current(i, y_current);
			if (axis.axis_ == Axis::X) {
				map_[current].vx = axis.map_[current];
			}
			else {
				map_[current].vy = axis.map_[current];
			}
		}
	}
}