#pragma once

#include "fluid_sim.hpp"

#include <vector>
#include <string>

#include <openvdb/openvdb.h>

using std::vector;
using std::string;

struct OpenVDBHandler {
	OpenVDBHandler() = default;
	OpenVDBHandler(FluidSim& sim, string file_name = "FluidSim");

	openvdb::GridPtrVec CreateGrids();
	vector<openvdb::FloatGrid::Accessor> GetAccessors();

	void LoadData();

	void WriteFile();

	void LoadData(AxisData& v_x, AxisData& v_y, AxisData& v_z, AxisData& density);
	void WriteFile(AxisData& v_x, AxisData& v_y, AxisData& v_z, AxisData& density);

	FluidSim sim_;
	vector<openvdb::FloatGrid::Ptr> grids_;
	vector<string> names_;

	string file_name_;

private:
	unsigned int index_ = 0;
	openvdb::GridPtrVec grid_vec;
	void FreeFieldPointers(openvdb::GridPtrVec grid_vec);
};