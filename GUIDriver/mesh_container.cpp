#include "vertex_data.hpp"

MeshContainer::MeshContainer(const bool& collision_mode, const bool& sync_mode_in) {
    collision = collision_mode;
    sync_mode_ = sync_mode_in;
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

inline void SyncModeWarning() {
    ProgramLog::OutputLine("Warning: Sync Mode is enabled. This means non-pointer and dynamic resize operations are disabled!");
}

inline void SyncModeOperationWarning() {
    ProgramLog::OutputLine("Warning: Sync Mode is enabled. This means wrapped vector access or modifier operations are disabled!");
}

void MeshContainer::Push(std::initializer_list<Vertex>& coords_in) {
    if (sync_mode_) {
        SyncModeWarning();
        return;
    }
    for (auto coord_in : coords_in) {
        Push(coord_in);
    }
}

bool MeshContainer::CollisionCheck(Vertex& coord_in) {
    if (sync_mode_) {
        SyncModeWarning();
        return false;
    }
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
    if (sync_mode_) {
        SyncModeOperationWarning();
        Vertex vertex(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        return vertex;
    }

    if (index < 0 || index >= vertices_.size()) {
        ProgramLog::OutputLine("Warning: Out of bounds access on mesh container!");
        return vertices_[0];
    }
    return vertices_[index];
}

void MeshContainer::Set(Vertex& index_coord, const unsigned int& index) {
    if (sync_mode_) {
        SyncModeOperationWarning();
        return;
    }
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
    if (sync_mode_) {
        SyncModeOperationWarning();
        return;
    }
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
    if (sync_mode_) { //NOTE: This could be supported later for sync mode
        SyncModeOperationWarning();
        return;
    }
    vertices_.clear();
    if (collision) {
        map_.clear();
    }
    ProgramLog::OutputLine("Cleared mesh data successfully!");
}

const Vertex* MeshContainer::Data() {
    if (sync_mode_) {
        if (sync_data_.size == 0) {
            ProgramLog::OutputLine("Warning: No vertices stored!");
        }
        return (Vertex*) sync_data_.cuda_device_ptr; //NOTE: Maybe an incorrect cast? Just a note, also, might not be necessitated
    }
    if (vertices_.size() == 0) {
        ProgramLog::OutputLine("Warning: No vertices stored!");
    }
    return vertices_.data();
}

unsigned int MeshContainer::Size() {
    return vertices_.size();
}

bool MeshContainer::SyncMode() const {
    return sync_mode_;
}