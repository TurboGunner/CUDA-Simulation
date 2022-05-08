#pragma once

#include <string>

using std::string;

struct IndexPair {
	/// <summary> 
	/// Default constructor, each dimension's default assignment is 0.
	/// </summary>
	IndexPair() = default;

	/// <summary> 
	/// Loaded constructor, takes in an unsigned int for each dimension.
	/// </summary>
	IndexPair(unsigned int x_in, unsigned int y_in);

	/// <summary> 
	/// Operator overload for copying the data of an existing index pair.
	/// </summary>
	bool operator==(const IndexPair& i1) const;

	/// <summary> 
	/// Operator overload for returning the proper hash code for IndexPair.
	/// </summary>
	size_t operator()(const IndexPair& i1) const noexcept;

	/// <summary> 
	/// Operator overload for comparing (less than). Required for proper functioning with maps.
	/// </summary>
	bool operator<(const IndexPair& i1) const;

	/// <summary> 
	/// Returns an std::string of the coordinate values of IndexPair.
	/// </summary>
	string ToString() const;

	unsigned int x = 0, y = 0; //Spots
};