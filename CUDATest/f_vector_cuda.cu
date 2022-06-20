#include "f_vector_cuda.cuh"

#include <math.h>

__host__ __device__ F_Vector::F_Vector(float x_in, float y_in) { //Loaded Constructor
	vx_ = x_in;
	vy_ = y_in;
}

float* F_Vector::AsArrPair() { //Output as float pointer
	float output[2] = { vx_, vy_ };
	return output;
}

float F_Vector::Magnitude() const {
	float combined = pow(vx_, 2) + pow(vy_, 2);
	return sqrt(combined);
}

bool F_Vector::operator==(const F_Vector& v1) const {
	return vx_ == v1.vx_ && vy_ == v1.vy_;
}


__host__ __device__ const F_Vector F_Vector::operator+(const F_Vector& v1) const {
	F_Vector output = *this;
	output.vx_ += v1.vx_;
	output.vy_ += v1.vy_;
	return output;
}

__host__ __device__ const F_Vector F_Vector::operator-(const F_Vector& v1) const {
	F_Vector output = *this;
	output.vx_ -= v1.vx_;
	output.vy_ -= v1.vy_;
	return output;
}

__host__ __device__ F_Vector F_Vector::operator*(float num) {
	vx_ *= num;
	vy_ *= num;
	return *this;
}

__host__ __device__ F_Vector F_Vector::operator*(unsigned int num) {
	vx_ *= num;
	vy_ *= num;
	return *this;
}

size_t F_Vector::operator()(const F_Vector& v1) const noexcept {
	size_t hash1 = std::hash<float>()(v1.vx_);
	size_t hash2 = std::hash<float>()(v1.vy_);
	return hash1 ^ (hash2 << 1);
}

__host__ __device__ void F_Vector::operator=(const F_Vector& copy) {
	vx_ = copy.vx_;
	vy_ = copy.vy_;
}

string F_Vector::ToString() const {
	return "X Component: " + std::to_string(vx_) + " | Y Component: " + std::to_string(vy_);
}