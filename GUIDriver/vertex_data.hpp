#pragma once

#include "cross_memory_handle.cuh"

#include "vulkan_helpers.hpp"

#include "../CUDATest/handler_classes.hpp"

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include <array>
#include <vector>
#include <unordered_map>
#include <cmath>

using std::array;
using std::vector;
using std::unordered_map;

struct VectorHash {
    size_t operator() (const glm::vec3& vector) const {
        size_t h1 = std::hash<float>()(vector.x);
        size_t h2 = std::hash<float>()(vector.y);
        size_t h3 = std::hash<float>()(vector.z);

        return h1 ^ h2 ^ h3;
    }
};

struct Vertex {
    __host__ __device__ Vertex() = default;

    __host__ __device__ Vertex(float pos_x, float pos_y, float pos_z, float r, float g, float b, float n_x = 0.0f, float n_y = 0.0f, float n_z = 0.0f);

    void RGBAdjust(float r, float g, float b, float size = 255.0f);

    static vector<VkVertexInputBindingDescription> GetBindingDescription();

    static vector<VkVertexInputAttributeDescription> GetAttributeDescriptions();

    Vertex& operator=(const Vertex& copy);

    bool operator==(Vertex const& compare);

    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 color;
};


class MeshContainer {
public:
    MeshContainer(const bool& collision_mode = false, const bool& sync_mode_in = true);

    void Push(Vertex& coord_in);

    void Push(vector<Vertex>& coords_in);

    void Push(std::initializer_list<Vertex>& coords_in);

    bool CollisionCheck(Vertex& coord_in);

    Vertex& operator[](const int& index);

    Vertex& Get(const int& index);

    void Set(Vertex& index_coord, const unsigned int& index);

    void Remove(const unsigned int& index);

    void Clear();

    const Vertex* Data();

    unsigned int Size();

    bool SyncMode() const;

    CrossMemoryHandle sync_data_;

private:
    vector<Vertex> vertices_;

    bool sync_mode_ = true;

    unordered_map<glm::vec3, unsigned int, VectorHash> map_;
    bool collision = false;
};