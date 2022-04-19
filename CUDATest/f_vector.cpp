#include "vector_field.hpp"

F_Vector::F_Vector(float x_in, float y_in) { //Loaded Constructor
	vx = x_in;
	vy = y_in;
}

float* F_Vector::AsArrPair() { //Output as float pointer
	float output[2] = { vx, vy };
	return output;
}

float F_Vector::Magnitude() const {
	float combined = pow(vx, 2) + pow(vy, 2);
	return sqrt(combined);
}

bool F_Vector::operator==(const F_Vector& v1) const {
	return vx == v1.vx && vy == v1.vy;
}

const F_Vector F_Vector::operator+(const F_Vector& v1) const {
	F_Vector output = *this;
	output.vx += v1.vx;
	output.vy += v1.vy;
	return output;
}

const F_Vector F_Vector::operator-(const F_Vector& v1) const {
	F_Vector output = *this;
	output.vx -= v1.vx;
	output.vy -= v1.vy;
	return output;
}

F_Vector F_Vector::operator*(float num) {
	vx *= num;
	vy *= num;
	return *this;
}

F_Vector F_Vector::operator*(unsigned int num) {
	vx *= num;
	vy *= num;
	return *this;
}

size_t F_Vector::operator()(const F_Vector& v1) const noexcept {
	size_t hash1 = std::hash<float>()(v1.vx);
	size_t hash2 = std::hash<float>()(v1.vy);
	return hash1 ^ (hash2 << 1);
}

void F_Vector::operator=(const F_Vector& copy) {
	vx = copy.vx;
	vy = copy.vy;
}

string F_Vector::ToString() const {
	return "X Component: " + std::to_string(vx) + " | Y Component: " + std::to_string(vy);
}