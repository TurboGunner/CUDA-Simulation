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

	FluidSim sim_;
	vector<openvdb::FloatGrid::Ptr> grids_;
	vector<string> names_;

	string file_name_;

private:
	unsigned int index_ = 0;
	void FreeFieldPointers(openvdb::GridPtrVec grid_vec);
};