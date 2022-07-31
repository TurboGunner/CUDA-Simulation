#include "openvdb_handler.hpp"

#include <format>
#include <iostream>
#include <stdexcept>

OpenVDBHandler::OpenVDBHandler(FluidSim& sim, string file_name) {
	sim_ = sim;
	names_.insert(names_.end(), {"Density"});
	file_name_ = file_name;

	openvdb::initialize();
	grid_vec = CreateGrids();

	std::cout << sim_.density_.map_->Get(IndexPair(32, 32, 32).IX(sim_.size_.x)) << std::endl;
}

OpenVDBHandler::~OpenVDBHandler() {
	FreeFieldPointers();
}

openvdb::GridPtrVec OpenVDBHandler::CreateGrids() {
	openvdb::GridPtrVec grids;

	openvdb::FloatGrid::Ptr grid_x = openvdb::FloatGrid::create(),
		grid_y = openvdb::FloatGrid::create(),
		grid_z = openvdb::FloatGrid::create(),
		grid_density = openvdb::FloatGrid::create();

	grid_density->setName(names_.at(0));
	grid_density->setTransform(transform_);

	grids.insert(grids.end(), { grid_density });
	grids_.insert(grids_.end(), { grid_density });
	return grids;
}

vector<openvdb::FloatGrid::Accessor> OpenVDBHandler::GetAccessors() {
	vector<openvdb::FloatGrid::Accessor> accessors;

	openvdb::FloatGrid::Accessor accessor_density = grids_.at(0)->getAccessor();

	accessors.insert(accessors.end(), { accessor_density });
	return accessors;
}

void OpenVDBHandler::LoadData() {
	LoadData(sim_.density_);
}

void OpenVDBHandler::LoadData(AxisData& density) {
	vector<openvdb::FloatGrid::Accessor> accessors = GetAccessors();
	unsigned int y_current = 0, z_current = 0;

	openvdb::Coord xyz(0, 0, 0);

	for (z_current; z_current < sim_.size_.z; z_current++) {
		for (y_current; y_current < sim_.size_.y; y_current++) {
			for (unsigned int i = 0; i < sim_.size_.x; i++) {
				IndexPair current(i, y_current, z_current);
				xyz.reset(i, y_current, z_current);
				accessors.at(0).setValue(xyz,
					density.map_->Get(IndexPair(i, y_current, z_current).IX(sim_.size_.x)));
			}
		}
	}
}

void OpenVDBHandler::WriteFile() {
	WriteFile(sim_.density_);
}

void OpenVDBHandler::WriteFile(AxisData& density) {
	string file_extension = file_name_ + std::format("{:04}", index_) + ".vdb";
	openvdb::io::File file(file_extension);

	LoadData(density);

	file.write(grid_vec);
	file.close();
	index_++;
}

void OpenVDBHandler::FreeFieldPointers() {
	for (int i = 0; i < grid_vec.size(); i++) {
		grid_vec.at(i).~shared_ptr();
	}
}