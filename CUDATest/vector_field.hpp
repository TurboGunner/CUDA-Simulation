#pragma once

#include <math.h>

#include <string>
#include <set>
#include <map>

using std::string;
using std::set;
using std::map;

struct F_Vector {
	F_Vector(float x_in = 1, float y_in = 1) { //Loaded Constructor
		vx = x_in;
		vy = y_in;
	}

	float* AsArrPair() { //Output as float pointer
		float output[2] = { vx, vy };
		return output;
	}

	float Magnitude() const {
		float combined = pow(vx, 2) + pow(vy, 2);
		return sqrt(combined);
	}

	bool operator==(const F_Vector& v1) const {
		return vx == v1.vx && vy == v1.vy;
	}

	const F_Vector operator+(const F_Vector& v1) const {
		F_Vector output = *this;
		output.vx += v1.vx;
		output.vy += v1.vy;
		return output;
	}

	const F_Vector operator-(const F_Vector& v1) const {
		F_Vector output = *this;
		output.vx -= v1.vx;
		output.vy -= v1.vy;
		return output;
	}

	F_Vector operator*(float num) {
		vx *= num;
		vy *= num;
		return *this;
	}

	F_Vector operator*(unsigned int num) {
		vx *= num;
		vy *= num;
		return *this;
	}

	size_t operator()(const F_Vector& v1) const noexcept {
		size_t hash1 = std::hash<float>()(v1.vx);
		size_t hash2 = std::hash<float>()(v1.vy);
		return hash1 ^ (hash2 << 1);
	}

	void operator=(const F_Vector& copy) {
		vx = copy.vx;
		vy = copy.vy;
	}

	string ToString() const {
		return "X Component: " + std::to_string(vx) +  " | Y Component: " + std::to_string(vy);
	}

	float vx, vy; //Components
};

struct IndexPair {
	IndexPair() = default;
	IndexPair(unsigned int x_in, unsigned int y_in) { //Loaded Constructor
		x = x_in;
		y = y_in;
	}

	bool operator==(const IndexPair& i1)  const {
		return x == i1.x && y == i1.y;
	}

	unsigned int operator()(const IndexPair& i1) const noexcept {
		unsigned int hash1 = std::hash<unsigned int>()(i1.x);
		unsigned int hash2 = std::hash<unsigned int>()(i1.y);
		return hash1 ^ (hash2 << 1);
	}

	bool operator<(const IndexPair& i1)  const {
		if (y == i1.y) {
			return x < i1.x;
		}
		return y < i1.y;
	}

	string ToString() const {
		return "X Component: " + std::to_string(x) + " | Y Component: " + std::to_string(y);
	}

	unsigned int x = 0, y = 0; //Spots
};

class VectorField {
	public:
		//Constructor
		VectorField(unsigned int x = 1, unsigned int y = 1);
		VectorField(unsigned int x, unsigned int y, const set<F_Vector>& set);

		//Accessor Methods
		map<IndexPair, F_Vector>& GetVectorMap() { //Reference
			return map_;
		}

		unsigned int GetSizeX() const {
			return size_x_;
		}
		unsigned int GetSizeY() const {
			return size_y_;
		}

		void operator=(const VectorField& copy) {
			map_ = copy.map_;
		}

		void SetSizeX(unsigned int x) {
			size_x_ = x;
		}
		void SetSizeY(unsigned int y) {
			size_y_ = y;
		}

		string ToString();

	private:
		map<IndexPair, F_Vector> map_;
		unsigned int size_x_, size_y_;
		set<F_Vector> LoadDefaultVectorSet();
};