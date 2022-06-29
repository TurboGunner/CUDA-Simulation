#include "openvdb_handler.hpp"

#include <format>
#include <iostream>
#include <stdexcept>

OpenVDBHandler::OpenVDBHandler(FluidSim& sim, string file_name) {
	sim_ = sim;
	names_.insert(names_.end(), {"VelocityVectorX", "VelocityVectorY", "VelocityVectorZ", "Density"});
	file_name_ = file_name;

	openvdb::initialize();
	grid_vec = CreateGrids();

	std::cout << sim_.density_.map_->Get(IndexPair(32, 32, 32).IX(sim_.size_.x)) << std::endl;
}

openvdb::GridPtrVec OpenVDBHandler::CreateGrids() {
	openvdb::GridPtrVec grids;

	openvdb::FloatGrid::Ptr grid_x = openvdb::FloatGrid::create(),
		grid_y = openvdb::FloatGrid::create(),
		grid_z = openvdb::FloatGrid::create(),
		grid_density = openvdb::FloatGrid::create();

	grid_x->setName(names_.at(0));
	grid_y->setName(names_.at(1));
	grid_z->setName(names_.at(2));
	grid_density->setName(names_.at(3));

	grids.insert(grids.end(), { grid_x, grid_y, grid_z, grid_density });
	grids_.insert(grids_.end(), { grid_x, grid_y, grid_z, grid_density });
	return grids;
}

vector<openvdb::FloatGrid::Accessor> OpenVDBHandler::GetAccessors() {
	vector<openvdb::FloatGrid::Accessor> accessors;

	openvdb::FloatGrid::Accessor accessor_x = grids_.at(0)->getAccessor(),
		accessor_y = grids_.at(1)->getAccessor(),
		accessor_z = grids_.at(2)->getAccessor(),
		accessor_density = grids_.at(3)->getAccessor();

	accessors.insert(accessors.end(), { accessor_x, accessor_y, accessor_z, accessor_density });
	return accessors;
}

void OpenVDBHandler::LoadData() {
	LoadData(sim_.velocity_.map_[0].map_,
		sim_.velocity_.map_[1].map_,
		sim_.velocity_.map_[2].map_,
		sim_.density_.map_);
}

void OpenVDBHandler::LoadData(HashMap<float>*& v_x, HashMap<float>*& v_y, HashMap<float>*& v_z, HashMap<float>*& density) {
	vector<openvdb::FloatGrid::Accessor> accessors = GetAccessors();
	unsigned int y_current = 0, z_current = 0;

	openvdb::Coord xyz(0, 0, 0);

	for (z_current; z_current < sim_.size_.z; z_current++) {
		for (y_current; y_current < sim_.size_.y; y_current++) {
			for (unsigned int i = 0; i < sim_.size_.x; i++) {
				IndexPair current(i, y_current, z_current);
				xyz.reset(i, y_current, z_current);
				accessors.at(0).setValue(xyz, v_x->Get(current.IX(sim_.size_.x)));
				accessors.at(1).setValue(xyz, v_y->Get(current.IX(sim_.size_.x)));
				accessors.at(2).setValue(xyz, v_z->Get(current.IX(sim_.size_.x)));
				accessors.at(3).setValue(xyz, density->Get(IndexPair(31, 31, 31).IX(sim_.size_.x)));
			}
		}
	}
}

void OpenVDBHandler::WriteFile() {
	WriteFile(sim_.velocity_.map_[0].map_,
		sim_.velocity_.map_[1].map_,
		sim_.velocity_.map_[2].map_,
		sim_.density_.map_);
}

void OpenVDBHandler::WriteFile(HashMap<float>*& v_x, HashMap<float>*& v_y, HashMap<float>*& v_z, HashMap<float>*& density) {
	string file_extension = file_name_ + std::format("{:04}", index_) + ".vdb";
	openvdb::io::File file(file_extension);

	LoadData(v_x, v_y, v_z, density);

	file.write(grid_vec);
	file.close();
	index_++;
}

void OpenVDBHandler::FreeFieldPointers(openvdb::GridPtrVec grid_vec) {
	for (int i = 0; i < grid_vec.size(); i++) {
		grid_vec.at(i).~shared_ptr();
	}
}