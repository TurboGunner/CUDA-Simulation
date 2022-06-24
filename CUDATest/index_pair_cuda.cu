#include "index_pair_cuda.cuh"

__host__ __device__ IndexPair::IndexPair(unsigned int x_in, unsigned int y_in, unsigned int z_in) { //Loaded Constructor
	x = x_in;
	y = y_in;
	z = z_in;
}
__host__ __device__ size_t IndexPair::IX(size_t size) const {
	return x + (y * size);
}


bool IndexPair::operator==(const IndexPair& i1) const {
	return x == i1.x && y == i1.y;
}

size_t IndexPair::operator()(const IndexPair& i1) const noexcept {
	unsigned int hash1 = std::hash<unsigned int>()(i1.x);
	unsigned int hash2 = std::hash<unsigned int>()(i1.y);
	return hash1 ^ (hash2 << 1);
}

bool IndexPair::operator<(const IndexPair& i1) const {
	if (y == i1.y) {
		return x < i1.x;
	}
	return y < i1.y;
}

string IndexPair::ToString() const {
	return "X Component: " + std::to_string(x) + " | Y Component: " + std::to_string(y);
}

__host__ __device__ IndexPair IndexPair::Left() {
	return IndexPair(x - 1, y, z);
}

__host__ __device__ IndexPair IndexPair::Right() {
	return IndexPair(x + 1, y, z);
}

__host__ __device__ IndexPair IndexPair::Up() {
	return IndexPair(x, y + 1, z);
}

__host__ __device__ IndexPair IndexPair::Down() {
	return IndexPair(x, y - 1, z);
}

__host__ __device__ IndexPair IndexPair::Front() {
	return IndexPair(x, y, z + 1);
}

__host__ __device__ IndexPair IndexPair::Back() {
	return IndexPair(x, y, z - 1);
}