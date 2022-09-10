#pragma once

#include "vulkan_helpers.hpp"

#include "../CUDATest/handler_classes.hpp"

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include <array>
#include <vector>
#include <unordered_map>

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
    Vertex() = default;

    Vertex(float pos_x, float pos_y, float pos_z, float r, float g, float b, float n_x = 0.0f, float n_y = 0.0f, float n_z = 0.0f) {
        pos = glm::vec3(pos_x, pos_y, pos_z);
        color = glm::vec3(r, g, b);
        glm::vec3(n_x, n_y, n_z);
    }

    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 color;

    static vector<VkVertexInputBindingDescription> GetBindingDescription() {
        VkVertexInputBindingDescription binding_description = {};

        binding_description.binding = 0;
        binding_description.stride = sizeof(Vertex);
        binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return { binding_description };
    }

    static vector<VkVertexInputAttributeDescription> GetAttributeDescriptions() {
        vector<VkVertexInputAttributeDescription> attribute_descriptions;

        VkVertexInputAttributeDescription pos_description = {};
        pos_description.binding = 0;
        pos_description.location = 0;
        pos_description.format = VK_FORMAT_R32G32_SFLOAT;
        pos_description.offset = offsetof(Vertex, pos);

        VkVertexInputAttributeDescription normal_description = {};
        normal_description.binding = 0;
        normal_description.location = 1;
        normal_description.format = VK_FORMAT_R32G32B32_SFLOAT;
        normal_description.offset = offsetof(Vertex, normal);

        VkVertexInputAttributeDescription color_description = {};
        color_description.binding = 0;
        color_description.location = 2;
        color_description.format = VK_FORMAT_R32G32B32_SFLOAT;
        color_description.offset = offsetof(Vertex, color);

        attribute_descriptions.insert(attribute_descriptions.end(), { pos_description, normal_description, color_description });

        return attribute_descriptions;
    }

    Vertex& operator=(const Vertex& copy) {
        pos = copy.pos;
        color = copy.color;

        return *this;
    }

    bool operator==(Vertex const& compare) {
        bool pos_check = pos.x == compare.pos.x && pos.y == compare.pos.y && pos.z == compare.pos.z;
        bool color_check = color.x == compare.color.x && color.y == compare.color.y && color.z == compare.color.z;
        bool normal_check = normal.x == compare.normal.x && normal.y == compare.normal.y && normal.z == compare.normal.z;

        return pos_check && color_check && normal_check;
    }
};


class MeshContainer {
public:
    MeshContainer() = default;

    MeshContainer(bool collision_mode) {
        collision = collision_mode;
    }

    void Push(Vertex& coord_in) {
        if (!collision) {
            vertices_.push_back(coord_in);
            return;
        }
        bool collide = CollisionCheck(coord_in);
        if (!collide) {
            vertices_.push_back(coord_in);
        }
    }

    void Push(vector<Vertex>& coords_in) {
        for (auto coord_in : coords_in) {
            Push(coord_in);
        }
    }

    void Push(std::initializer_list<Vertex>& coords_in) {
        for (auto coord_in : coords_in) {
            Push(coord_in);
        }
    }

    bool CollisionCheck(Vertex& coord_in) {
        size_t original_size = vertices_.size();
        map_.try_emplace(coord_in.pos, vertices_.size());
        if (vertices_.size() == original_size) {
            ProgramLog::OutputLine("Warning: Intersecting vertex!");
            return true;
        }
        return false;
    }

    Vertex& operator[](const int& index) {
        return Get(index);
    }

    Vertex& Get(const int& index) {
        if (index < 0 || index >= vertices_.size()) {
            ProgramLog::OutputLine("Warning: Out of bounds access on mesh container!");
            return vertices_[0];
        }
        return vertices_[index];
    }

    void Set(Vertex& index_coord, const unsigned int& index) {
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

    void Remove(const unsigned int& index) {
        if (index >= vertices_.size()) {
            ProgramLog::OutputLine("Warning: Out of bounds access on mesh container!");
            return;
        }
        if (collision) {
            map_.erase(vertices_[index].pos);
        }
        vertices_.erase(vertices_.begin() + index);
    }

    void Clear() {
        vertices_.clear();
        if (collision) {
            map_.clear();
        }
        ProgramLog::OutputLine("Cleared mesh data successfully!");
    }

    const Vertex* Data() {
        if (vertices_.size() == 0) {
            ProgramLog::OutputLine("Warning: No vertices stored!");
        }
        return vertices_.data();
    }

    unsigned int Size() {
        return vertices_.size();
    }

private:
    vector<Vertex> vertices_;
    unordered_map<glm::vec3, unsigned int, VectorHash> map_;
    bool collision = false;
};