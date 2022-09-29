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

    VertexData(VkDevice& device_in, VkPhysicalDevice& phys_device_in, VkQueue& queue_in, const size_t& max_frames_const_in) {
        device_ = device_in;
        physical_device_ = phys_device_in;
        queue_ = queue_in;

        vector<Vertex> vertices_in = {
            Vertex(0.5f, -0.5f, 0.0f, 1.0f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f),
            Vertex(0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 1.0f, 0.5f, 0.5f, 0.5f),
            Vertex(-1.5f, 1.0f, 1.0f, 1.0f, 0.5f, 1.0f, 0.5f, 0.5f, 0.5f),
            Vertex(-1.5f, -1.5f, -1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f),
            Vertex(1.0f, -1.5f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f),
            Vertex(1.0f, 1.0f, 1.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f)
        };

        vertices.Push(vertices_in);
        size_ = sizeof(vertices[0]) * vertices.Size();

        MAX_FRAMES_IN_FLIGHT_ = max_frames_const_in;
    }

    void BindPipeline(VkCommandBuffer& command_buffer, VkCommandPool& command_pool) {
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer_, offsets);
        vkCmdBindIndexBuffer(command_buffer, index_buffer_, 0, VK_INDEX_TYPE_UINT16);
    }

    void Initialize(VkCommandPool& command_pool) {
        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_memory;

        //Staging Buffer
        BufferHelpers::CreateBuffer(device_, physical_device_, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, size_, staging_buffer, staging_buffer_memory);

        void* data = BufferHelpers::MapMemory(device_, vertices.Data(), size_, staging_buffer_memory);

        //Vertex Buffer
        BufferHelpers::CreateBuffer(device_, physical_device_, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, size_, vertex_buffer_, vertex_buffer_memory_);

        BufferHelpers::CopyBuffer(device_, queue_, command_pool, staging_buffer, vertex_buffer_, size_);

        //Destroying Staging Buffer
        vkDestroyBuffer(device_, staging_buffer, nullptr);
        vkFreeMemory(device_, staging_buffer_memory, nullptr);
    }

    VkResult InitializeIndex(VkCommandPool& command_pool) {
        VkResult vulkan_status = VK_SUCCESS;

        const vector<uint16_t> indices = { 0, 1, 2, 0, 3, 2 }; //Make this a data member for the vert

        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_memory;

        vulkan_status = BufferHelpers::CreateBuffer(device_, physical_device_, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, size_, staging_buffer, staging_buffer_memory);

        void* data = BufferHelpers::MapMemory(device_, indices.data(), size_, staging_buffer_memory);

        vulkan_status = BufferHelpers::CreateBuffer(device_, physical_device_, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, size_, index_buffer_, index_buffer_memory_);

        BufferHelpers::CopyBuffer(device_, queue_, command_pool, staging_buffer, index_buffer_, size_);

        vkDestroyBuffer(device_, staging_buffer, nullptr);
        vkFreeMemory(device_, staging_buffer_memory, nullptr);

        return vulkan_status;
    }

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

    void Clean() {
        vkDestroyBuffer(device_, vertex_buffer_, nullptr);
        vkDestroyBuffer(device_, index_buffer_, nullptr);

        vkFreeMemory(device_, vertex_buffer_memory_, nullptr);
        vkFreeMemory(device_, index_buffer_memory_, nullptr);
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