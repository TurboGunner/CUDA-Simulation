#pragma once

#include "vulkan_helpers.hpp"
#include "vertex_data.hpp"

#include "../CUDATest/handler_classes.hpp"

#include <vulkan/vulkan.h>

class VertexData {
public:
    VertexData() = default;

    VertexData(VkDevice& device_in, VkPhysicalDevice& phys_device_in, VkQueue& queue_in) {
        device_ = device_in;
        physical_device_ = phys_device_in;
        queue_ = queue_in;

        //NOTE
        vector<Vertex> vertices_in = {
            Vertex(0.5f, -0.5f, 0.0f, 1.0f, 1.0f, 1.0f),
            Vertex(0.5f, 0.5f, 0.0f, 1.0f, 1.0f, 1.0f),
            Vertex(-0.5f, 0.5f, 0.0f, 1.0f, 1.0f, 1.0f)
        };
        vertices.Push(vertices_in);
    }

    void BindPipeline(VkCommandBuffer& command_buffer, VkCommandPool& command_pool) {
        VkDeviceSize size = sizeof(vertices[0]) * vertices.Size();
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer_, &size);
    }

    void Initialize(VkCommandPool& command_pool) {
        VkDeviceSize size = sizeof(vertices[0]) * vertices.Size();

        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_memory;

        //Staging Buffer
        CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, size, staging_buffer, staging_buffer_memory);

        MapMemory(size, staging_buffer_memory);

        //Vertex Buffer
        CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, size, vertex_buffer_, vertex_buffer_memory_);

        CopyBuffer(command_pool, staging_buffer, vertex_buffer_, size);

        //Destroying Staging Buffer
        vkDestroyBuffer(device_, staging_buffer, nullptr);
        vkFreeMemory(device_, staging_buffer_memory, nullptr);
    }

    void CreateBuffer(const VkBufferUsageFlags& usage, const VkMemoryPropertyFlags& properties, const VkDeviceSize& size, VkBuffer& buffer, VkDeviceMemory& buffer_memory) {
        VkBufferCreateInfo buffer_info = CreateVertexBuffer(size, usage);

        if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed to properly allocate buffer!");
        }

        VkMemoryRequirements mem_requirements;
        vkGetBufferMemoryRequirements(device_, buffer, &mem_requirements);

        VkMemoryAllocateInfo alloc_info = VulkanHelper::CreateAllocationInfo(physical_device_, mem_requirements, properties);

        if (vkAllocateMemory(device_, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device_, buffer, buffer_memory, 0);

        ProgramLog::OutputLine("Created and bound buffer to device memory successfully!");
    }

    void* MapMemory(const VkDeviceSize& size, VkDeviceMemory& device_memory) {
        void* data;

        vkMapMemory(device_, device_memory, 0, size, 0, &data);
        memcpy(data, vertices.Data(), size);
        vkUnmapMemory(device_, device_memory);

        return data;
    }

    void Clean() {
        vkDestroyBuffer(device_, vertex_buffer_, nullptr);
        vkFreeMemory(device_, vertex_buffer_memory_, nullptr);
    }

    MeshContainer vertices;

private:
    VkBufferCreateInfo CreateVertexBuffer(const VkDeviceSize& size, const VkBufferUsageFlags& usage) {
        VkBufferCreateInfo buffer_info = {};

        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size;
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        return buffer_info;
    }

    void CopyBuffer(VkCommandPool& command_pool, VkBuffer& src_buffer, VkBuffer& dst_buffer, const VkDeviceSize& size) {
        VkCommandBuffer command_buffer = VulkanHelper::BeginSingleTimeCommands(device_, command_pool);

        VkBufferCopy copyRegion = {};
        copyRegion.size = size;
        vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copyRegion);

        VulkanHelper::EndSingleTimeCommands(command_buffer, device_, command_pool, queue_);
    }

    VkDevice device_;
    VkPhysicalDevice physical_device_;
    VkQueue queue_;

    VkBuffer vertex_buffer_;
    VkDeviceMemory vertex_buffer_memory_;
};