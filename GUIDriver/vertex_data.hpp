#pragma once

#include "vulkan_helpers.hpp"

#include "../CUDATest/handler_classes.hpp"

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include <array>
#include <vector>

using std::array;
using std::vector;

struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription GetBindingDescription() {
        VkVertexInputBindingDescription binding_description = {};

        binding_description.binding = 0;
        binding_description.stride = sizeof(Vertex);
        binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return binding_description;
    }

    static array<VkVertexInputAttributeDescription, 2> GetAttributeDescriptions() {
        array<VkVertexInputAttributeDescription, 2> attribute_descriptions {};

        attribute_descriptions[0].binding = 0;
        attribute_descriptions[0].location = 0;
        attribute_descriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attribute_descriptions[0].offset = offsetof(Vertex, pos);

        attribute_descriptions[1].binding = 0;
        attribute_descriptions[1].location = 1;
        attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribute_descriptions[1].offset = offsetof(Vertex, color);

        return attribute_descriptions;
    }
};

class VertexData {
public:
    VertexData() = default;

    VertexData(VkDevice device_in, VkPhysicalDevice& phys_device_in) {
        device_ = device_in;
        physical_device_ = phys_device_in;
    }

    void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& buffer_memory) {
        VkBufferCreateInfo buffer_info = CreateVertexBuffer();

        if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed to properly allocate buffer!");
        }

        VkMemoryRequirements mem_requirements;
        vkGetBufferMemoryRequirements(device_, buffer, &mem_requirements);

        VkMemoryAllocateInfo alloc_info = VulkanHelper::CreateAllocationInfo(physical_device_, mem_requirements, properties);

        if (vkAllocateMemory(device_, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed allocate buffery memory!");
        }

        vkBindBufferMemory(device_, buffer, buffer_memory, 0);
    }

    void* MapMemory(const VkBufferCreateInfo& buffer_info, VkDeviceMemory& device_memory, ) {
        void* data = nullptr;

        vkMapMemory(device_, device_memory, 0, buffer_info.size, 0, &data);
        memcpy(data, vertices.data(), (size_t) buffer_info.size);
        vkUnmapMemory(device_, device_memory);
    }

    void Clean() {
        vkDestroyBuffer(device_, vertex_buffer_, nullptr);
        vkFreeMemory(device_, vertex_buffer_memory_, nullptr);
    }

    vector<Vertex> vertices;

private:
    VkBufferCreateInfo CreateVertexBuffer() {
        VkBufferCreateInfo buffer_info = {};

        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = sizeof(vertices[0]) * vertices.size();

        return buffer_info;

    }

    VkDevice device_;
    VkPhysicalDevice physical_device_;

    VkBuffer vertex_buffer_;
    VkDeviceMemory vertex_buffer_memory_;
};