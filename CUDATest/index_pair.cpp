#include "vector_field.hpp"

IndexPair::IndexPair(unsigned int x_in, unsigned int y_in) { //Loaded Constructor
	x = x_in;
	y = y_in;
}

bool IndexPair::operator==(const IndexPair& i1) const {
	return x == i1.x && y == i1.y;
}

unsigned int IndexPair::operator()(const IndexPair& i1) const noexcept {
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