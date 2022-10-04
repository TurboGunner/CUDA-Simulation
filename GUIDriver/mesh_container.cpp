#include "vertex_data.hpp"

MeshContainer::MeshContainer(const bool& collision_mode) {
    collision = collision_mode;
}

void MeshContainer::Push(Vertex& coord_in) {
    if (!collision) {
        vertices_.push_back(coord_in);
        return;
    }
    bool collide = CollisionCheck(coord_in);
    if (!collide) {
        vertices_.push_back(coord_in);
    }
}

void MeshContainer::Push(vector<Vertex>& coords_in) {
    for (auto coord_in : coords_in) {
        Push(coord_in);
    }
}

void MeshContainer::Push(std::initializer_list<Vertex>& coords_in) {
    for (auto coord_in : coords_in) {
        Push(coord_in);
    }
}

bool MeshContainer::CollisionCheck(Vertex& coord_in) {
    size_t original_size = vertices_.size();
    map_.try_emplace(coord_in.pos, vertices_.size());
    if (vertices_.size() == original_size) {
        ProgramLog::OutputLine("Warning: Intersecting vertex!");
        return true;
    }
    return false;
}

Vertex& MeshContainer::operator[](const int& index) {
    return Get(index);
}

Vertex& MeshContainer::Get(const int& index) {
    if (index < 0 || index >= vertices_.size()) {
        ProgramLog::OutputLine("Warning: Out of bounds access on mesh container!");
        return vertices_[0];
    }
    return vertices_[index];
}

void MeshContainer::Set(Vertex& index_coord, const unsigned int& index) {
    if (index >= vertices_.size()) {
        ProgramLog::OutputLine("Warning: Out of bounds access on mesh container!");
        return;
    }
    CollisionCheck(index_coord);
    if (collision) {
        map_.erase(vertices_[index].pos);
        map_.try_emplace(index_coord.pos, index);
    }
    vertices_[index] = index_coord;
}

void MeshContainer::Remove(const unsigned int& index) {
    if (index >= vertices_.size()) {
        ProgramLog::OutputLine("Warning: Out of bounds access on mesh container!");
        return;
    }
    if (collision) {
        map_.erase(vertices_[index].pos);
    }
    vertices_.erase(vertices_.begin() + index);
}

void MeshContainer::Clear() {
    vertices_.clear();
    if (collision) {
        map_.clear();
    }
    ProgramLog::OutputLine("Cleared mesh data successfully!");
}

const Vertex* MeshContainer::Data() {
    if (vertices_.size() == 0) {
        ProgramLog::OutputLine("Warning: No vertices stored!");
    }
    return vertices_.data();
}

unsigned int MeshContainer::Size() {
    return vertices_.size();
}