#include "index_pair_cuda.cuh"

__host__ __device__ IndexPair::IndexPair(unsigned int x_in, unsigned int y_in, unsigned int z_in, float weight_in) { //Loaded Constructor
	x = x_in;
	y = y_in;
	z = z_in;

	weight = weight_in;
}
__host__ __device__ size_t IndexPair::IX(size_t size) const {
	return x + (y * size) + (z * (size * size));
}

__host__ __device__ size_t IndexPair::IX(uint3 size) const {
	return x + (y * size.x) + (z * (size.y * size.z));
}


bool IndexPair::operator==(const IndexPair& i1) const {
	return x == i1.x && y == i1.y;
}

__host__ size_t IndexPair::operator()(const IndexPair& i1) const noexcept {
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

//Inner Neighbor Weights

//X Direction

__host__ __device__ IndexPair IndexPair::Left() {
	return IndexPair(x - 1, y, z, 0.05556f);
}

__host__ __device__ IndexPair IndexPair::Right() {
	return IndexPair(x + 1, y, z, 0.05556f);
}

//Y Direction

__host__ __device__ IndexPair IndexPair::Front() {
	return IndexPair(x, y, z + 1, 0.05556f);
}

__host__ __device__ IndexPair IndexPair::Back() {
	return IndexPair(x, y, z - 1, 0.05556f);
}

//Z Direction

__host__ __device__ IndexPair IndexPair::Up() {
	return IndexPair(x, y + 1, z, 0.05556f);
}

__host__ __device__ IndexPair IndexPair::Down() {
	return IndexPair(x, y - 1, z, 0.05556f);
}

//Outer Neighbor Weights

//Front Left Corners

__host__ __device__ IndexPair IndexPair::CornerLUpFront() {
	return IndexPair(x - 1, y + 1, z + 1, 0.027778f);
}

__host__ __device__ IndexPair IndexPair::CornerLDownFront() {
	return IndexPair(x - 1, y + 1, z - 1, 0.027778f);
}

//Front Right Corners

__host__ __device__ IndexPair IndexPair::CornerRUpFront() {
	return IndexPair(x + 1, y + 1, z + 1, 0.027778f);
}

__host__ __device__ IndexPair IndexPair::CornerRDownFront() {
	return IndexPair(x - 1, y + 1, z + 1, 0.027778f);
}

//Back Left Corners

__host__ __device__ IndexPair IndexPair::CornerLUpBack() {
	return IndexPair(x - 1, y - 1, z + 1, 0.027778f);
}

__host__ __device__ IndexPair IndexPair::CornerLDownBack() {
	return IndexPair(x - 1, y - 1, z - 1, 0.027778f);
}

//Back Right Corners

__host__ __device__ IndexPair IndexPair::CornerRUpBack() {
	return IndexPair(x + 1, y - 1, z + 1, 0.027778f);
}

__host__ __device__ IndexPair IndexPair::CornerRDownBack() {
	return IndexPair(x + 1, y - 1, z - 1, 0.027778f);
}

//Front Mid Corners

__host__ __device__ IndexPair IndexPair::CornerLMidFront() {
	return IndexPair(x - 1, y + 1, z, 0.027778f);
}

__host__ __device__ IndexPair IndexPair::CornerRMidFront() {
	return IndexPair(x + 1, y + 1, z, 0.027778f);
}

//Back Mid Corners

__host__ __device__ IndexPair IndexPair::CornerLMidBack() {
	return IndexPair(x - 1, y - 1, z, 0.027778f);
}

__host__ __device__ IndexPair IndexPair::CornerRMidBack() {
	return IndexPair(x + 1, y - 1, z, 0.027778f);
}