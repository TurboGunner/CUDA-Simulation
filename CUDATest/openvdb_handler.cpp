#include "openvdb_handler.hpp"

#include <stdexcept>
#include <iostream>

OpenVDBHandler::OpenVDBHandler(FluidSim& sim, string file_name) {
	sim_ = sim;
	names_.insert(names_.end(), {"VelocityVectorX", "VelocityVectorY", "Density"});
	file_name_ = file_name;

	openvdb::initialize();
}

openvdb::GridPtrVec OpenVDBHandler::CreateGrids() {
	openvdb::GridPtrVec grids;

	openvdb::FloatGrid::Ptr grid_x = openvdb::FloatGrid::create(),
		grid_y = openvdb::FloatGrid::create(),
		grid_density = openvdb::FloatGrid::create();

	grid_x->setName(names_.at(0));
	grid_y->setName(names_.at(1));
	grid_density->setName(names_.at(2));

	grids.insert(grids.end(), { grid_x, grid_y, grid_density });
	grids_.insert(grids_.end(), { grid_x, grid_y, grid_density });
	return grids;
}

vector<openvdb::FloatGrid::Accessor> OpenVDBHandler::GetAccessors() {
	vector<openvdb::FloatGrid::Accessor> accessors;

	openvdb::FloatGrid::Accessor accessor_x = grids_.at(0)->getAccessor(),
		accessor_y = grids_.at(1)->getAccessor(),
		accessor_density = grids_.at(2)->getAccessor();

	accessors.insert(accessors.end(), { accessor_x, accessor_y, accessor_density });
	return accessors;
}

void OpenVDBHandler::LoadData() {
	VectorField velocity = sim_.velocity_;
	AxisData density = sim_.density_;

	vector<openvdb::FloatGrid::Accessor> accessors = GetAccessors();
	unsigned int y_current = 0;

	openvdb::Coord xyz(0, 0, 0);

	for (y_current; y_current < sim_.size_y_; y_current++) {
		for (unsigned int i = 0; i < sim_.size_x_; i++) {
			IndexPair current(i, y_current);
			xyz.reset(i, y_current, 0);
			accessors.at(0).setValue(xyz, (*velocity.GetVectorMap()[0].map_)[current.IX(sim_.size_x_)]);
			accessors.at(1).setValue(xyz, (*velocity.GetVectorMap()[1].map_)[current.IX(sim_.size_y_)]);
			accessors.at(2).setValue(xyz, density.map_->Get(current.IX(sim_.size_y_)));
		}
	}
}

void OpenVDBHandler::WriteFile() {
	string file_extension = file_name_ + std::to_string(index_) + ".vdb";
	openvdb::io::File file(file_extension);

	openvdb::GridPtrVec grid_vec = CreateGrids();
	LoadData();

	file.write(grid_vec);
	file.close();
	FreeFieldPointers(grid_vec);
}

void OpenVDBHandler::FreeFieldPointers(openvdb::GridPtrVec grid_vec) {
	for (int i = 0; i < grid_vec.size(); i++) {
		grid_vec.at(i).~shared_ptr();
	}
}