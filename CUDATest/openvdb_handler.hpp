#pragma once

#include "fluid_sim.hpp"

#include <vector>
#include <string>

#include <openvdb/openvdb.h>

using std::vector;
using std::string;

class OpenVDBHandler {
public:
	/// <summary> Default constructor. </summary>
	OpenVDBHandler() = default;

	~OpenVDBHandler();

	/// <summary> Loaded constructor, Takes in a FluidSim object address, and an optional std::string.
	/// <para> The string defaults to FluidSim as a name for the file extension. </para> </summary>
	OpenVDBHandler(FluidSim& sim, string file_name = "FluidSim");

	/// <summary> Initializes float grids for VDB output. </summary>
	openvdb::GridPtrVec CreateGrids();

	/// <summary> Returns the accessors that correspond to the initialized grids. </summary>
	vector<openvdb::FloatGrid::Accessor> GetAccessors();

	/// <summary> Loads data from provided FluidSim. </summary>
	void LoadData();

	/// <summary> Writes data from loaded data. </summary>
	void WriteFile();

	/// <summary> Overload of LoadData that takes in the density AxisData directly. 
	/// <para> This was added as a debugging measure due to data points being at one point inaccessible from FluidSim. </para> </summary>
	void LoadData(AxisData& density);

	/// <summary> Overload of WriteData that takes in the density AxisData directly. 
	/// <para> This was added as a debugging measure due to data points being at one point inaccessible from FluidSim. </para> </summary>
	void WriteFile(AxisData& density);

	/// <summary> Iteratively frees grid pointers from vector. </summary>
	void FreeFieldPointers();

	FluidSim sim_;
	vector<openvdb::FloatGrid::Ptr> grids_;
	vector<string> names_;

	string file_name_;

private:
	openvdb::GridPtrVec grid_vec;
	openvdb::math::Transform::Ptr transform_ = openvdb::math::Transform::createLinearTransform(2.0);
	
	unsigned int index_ = 0;
};