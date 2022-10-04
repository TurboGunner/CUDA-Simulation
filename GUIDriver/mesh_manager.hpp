#pragma once

#include "vulkan_helpers.hpp"
#include "vertex_data.hpp"
#include "shader_loader.cuh"
#include "buffer_helpers.hpp"

#include "../CUDATest/handler_classes.hpp"

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

struct MeshPushConstants {
    glm::vec4 data;
    glm::mat4 render_matrix;
};

class VertexData {
public:
    VertexData() = default;

    VertexData(VkDevice& device_in, VkPhysicalDevice& phys_device_in, VkQueue& queue_in, const size_t& max_frames_const_in);

    void BindPipeline(VkCommandBuffer& command_buffer, VkCommandPool& command_pool);

    void Initialize(VkCommandPool& command_pool);

    VkResult InitializeIndex(VkCommandPool& command_pool);

    void Clean();

    tuple<VkBuffer, VkDeviceMemory> UploadMesh(void* device_mesh_ptr, const VkDeviceSize& size) {
        VkResult vulkan_status = VK_SUCCESS;

        VkBuffer buffer_device;
        VkDeviceMemory device_memory;

        VkExternalMemoryBufferCreateInfo external_buffer_info = {};

        external_buffer_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        external_buffer_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;

        void* data = BufferHelpers::MapMemory(device_, device_mesh_ptr, size_, device_memory);

        return { buffer_device, device_memory };
    }

    MeshContainer vertices;

private:
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    VkQueue queue_;
    VkDeviceSize size_;

    VkBuffer vertex_buffer_, index_buffer_, mesh_buffer_;
    VkDeviceMemory vertex_buffer_memory_, index_buffer_memory_, mesh_buffer_memory_;

    size_t MAX_FRAMES_IN_FLIGHT_;
};