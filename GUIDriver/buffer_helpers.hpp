#pragma once

#include "vulkan_helpers.hpp"

#include <vulkan/vulkan.h>

struct BufferHelpers {
    static void* MapMemory(VkDevice& device, const void* arr, const VkDeviceSize& size, VkDeviceMemory& device_memory) {
        void* data;

        vkMapMemory(device, device_memory, 0, size, 0, &data);
        memcpy(data, arr, size);
        vkUnmapMemory(device, device_memory);

        return data;
    }

    static VkResult CopyBuffer(VkDevice& device, VkQueue& queue, VkCommandPool& command_pool, VkBuffer& src_buffer, VkBuffer& dst_buffer, const VkDeviceSize& size) {
        VkResult vulkan_status = VK_SUCCESS;
        VkCommandBuffer command_buffer = VulkanHelper::BeginSingleTimeCommands(device, command_pool, false);

        VkBufferCopy copy_region = {};
        copy_region.size = size;
        vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);

        vulkan_status = VulkanHelper::EndSingleTimeCommands(command_buffer, device, command_pool, queue, false);
        return vulkan_status;
    }

    static VkBuffer AllocateBuffer(VkDevice& device, const VkDeviceSize& size, const VkBufferUsageFlags& usage) {
        VkBuffer buffer;
        VkBufferCreateInfo buffer_info = {};

        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size;
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed to properly allocate buffer!");
        }
        return buffer;
    }

    static VkResult CreateBuffer(VkDevice& device, VkPhysicalDevice& physical_device, const VkBufferUsageFlags& usage, const VkMemoryPropertyFlags& properties, const VkDeviceSize& size, VkBuffer& buffer, VkDeviceMemory& buffer_memory) {
        buffer = BufferHelpers::AllocateBuffer(device, size, usage);

        VkMemoryRequirements mem_requirements;

        vkGetBufferMemoryRequirements(device, buffer, &mem_requirements);

        VkMemoryAllocateInfo alloc_info = VulkanHelper::CreateAllocationInfo(physical_device, mem_requirements, properties, false);

        if (vkAllocateMemory(device, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed to allocate buffer memory!");
        }
        return vkBindBufferMemory(device, buffer, buffer_memory, 0);
    }

    static VkWriteDescriptorSet WriteDescriptorSetInfo(VkDescriptorSet& descriptor_set, VkBuffer& buffer, const VkDescriptorType& descriptor_type, const size_t& range_size, const int& offset = 0) {
        VkDescriptorBufferInfo buffer_info = {};

        buffer_info.buffer = buffer;
        buffer_info.offset = offset;
        buffer_info.range = range_size;

        VkWriteDescriptorSet descriptor_write_info = {};

        descriptor_write_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_write_info.pNext = nullptr;

        descriptor_write_info.dstBinding = 0;
        descriptor_write_info.dstSet = descriptor_set;

        descriptor_write_info.descriptorCount = 1;
        descriptor_write_info.descriptorType = descriptor_type;
        descriptor_write_info.pBufferInfo = &buffer_info;

        return descriptor_write_info;
    }

    static size_t PadUniformBufferSize(VkPhysicalDeviceProperties& properties, const size_t& original_size) {
        size_t minUboAlignment = properties.limits.minUniformBufferOffsetAlignment;
        size_t alignedSize = original_size;
        if (minUboAlignment > 0) {
            alignedSize = (alignedSize + minUboAlignment - 1) & ~(minUboAlignment - 1);
        }
        return alignedSize;
    }

};