#pragma once

#include <string>
#include <set>
#include <map>

using std::string;
using std::set;
using std::map;

struct F_Vector {
	F_Vector(float x_in = 1, float y_in = 1);

	float* AsArrPair();

	float Magnitude() const;

	bool operator==(const F_Vector& v1) const;

	const F_Vector operator+(const F_Vector& v1) const;
	const F_Vector operator-(const F_Vector& v1) const;

	F_Vector operator*(float num);
	F_Vector operator*(unsigned int num);

	size_t operator()(const F_Vector& v1) const noexcept;

	void operator=(const F_Vector& copy);

	string ToString() const;

	float vx, vy; //Components
};

struct IndexPair {
	IndexPair() = default;
	IndexPair(unsigned int x_in, unsigned int y_in);

	bool operator==(const IndexPair& i1) const;

	unsigned int operator()(const IndexPair& i1) const noexcept;

	bool operator<(const IndexPair& i1) const;

	string ToString() const;

	unsigned int x = 0, y = 0; //Spots
};

class VectorField {
	public:
		//Constructor
		VectorField(unsigned int x = 1, unsigned int y = 1);
		VectorField(unsigned int x, unsigned int y, const set<F_Vector>& set);

		//Accessor Methods
		map<IndexPair, F_Vector>& GetVectorMap();


		void operator=(const VectorField& copy);

		void SetSizeX(unsigned int x) {
			size_x_ = x;
		}
		void SetSizeY(unsigned int y) {
			size_y_ = y;
		}

		float* FlattenMapX();
		float* FlattenMapY();

		void RepackMap(float* x, float* y);

		string ToString();

	private:
		map<IndexPair, F_Vector> map_;
		unsigned int size_x_, size_y_;
		set<F_Vector> LoadDefaultVectorSet();
};