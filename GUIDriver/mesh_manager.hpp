#pragma once

#include "cross_memory_handle.cuh"

#include "vulkan_helpers.hpp"
#include "vertex_data.hpp"
#include "buffer_helpers.hpp"

#include "../CUDATest/handler_classes.hpp"

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include <tuple>

using std::tuple;

struct MeshPushConstants {
    glm::vec4 data;
    glm::mat4 render_matrix;
};

class VertexData {
public:
    VertexData() = default;

    VertexData(VkDevice& device_in, VkPhysicalDevice& phys_device_in, VkQueue& queue_in, const size_t& max_frames_const_in, const bool& binding_mode_in = false);

    void BindPipeline(VkCommandBuffer& command_buffer, VkCommandPool& command_pool);

    VkResult Initialize(VkCommandPool& command_pool);

    VkResult InitializeIndex(VkCommandPool& command_pool);

    bool IndexBindingMode() const;

    void Clean();

    MeshContainer vertices = MeshContainer(false, true);

private:
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    VkQueue queue_;
    VkDeviceSize size_;

    VkBuffer vertex_buffer_, index_buffer_, mesh_buffer_;
    VkDeviceMemory vertex_buffer_memory_, index_buffer_memory_, mesh_buffer_memory_;

    bool index_binding_mode_ = false;

    size_t MAX_FRAMES_IN_FLIGHT_;
};