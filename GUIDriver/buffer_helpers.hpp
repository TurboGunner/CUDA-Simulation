#pragma once

#include "vulkan_helpers.hpp"

#include <vulkan/vulkan.h>

struct BufferHelpers {
    static void* MapMemory(VkDevice& device, const void* arr, const VkDeviceSize size, VkDeviceMemory& device_memory);

    static VkResult CopyBuffer(VkDevice& device, VkQueue& queue, VkCommandPool& command_pool, VkBuffer& src_buffer, VkBuffer& dst_buffer, const VkDeviceSize size);

    static VkBufferCreateInfo CreateBufferInfo(const VkDeviceSize size, const VkBufferUsageFlags usage, const VkSharingMode sharing_mode = VK_SHARING_MODE_EXCLUSIVE);

    static VkBuffer AllocateBuffer(VkDevice& device, const VkDeviceSize size, const VkBufferUsageFlags usage);

    static VkResult CreateBuffer(VkDevice& device, VkPhysicalDevice& physical_device, const VkBufferUsageFlags usage, const VkMemoryPropertyFlags properties, const VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& buffer_memory);

    static VkWriteDescriptorSet WriteDescriptorSetInfo(VkDescriptorSet& descriptor_set, VkBuffer& buffer, const VkDescriptorType& descriptor_type, const size_t range_size, const int offset = 0);

    static size_t PadUniformBufferSize(VkPhysicalDeviceProperties& properties, const size_t original_size);

    static VkResult CreateBufferCross(VkDevice& device, VkPhysicalDevice& physical_device, VkQueue& queue, VkCommandPool& command_pool, const void* ptr, VkBuffer& buffer, VkDeviceMemory& buffer_memory, const VkBufferUsageFlags usage_flags, const size_t size);
};