#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <vulkan/vulkan.h>

//Logging
#include "../CUDATest/handler_classes.hpp"

#include <tuple>

using std::tuple;

class TextureLoader {
public:
    TextureLoader() = default;
    TextureLoader(VkDevice& device_in, VkPhysicalDevice& physical_in, uint32_t family_in) {
        device_ = device_in; 
        physical_device_ = physical_in;

        image_format_ = VK_FORMAT_R8G8B8_UINT;

        ProgramLog::OutputLine("Initialized texture handler.\n");
    }

    ~TextureLoader() {
        vkDestroyImageView(device_, image_view_, nullptr);
        vkDestroyBuffer(device_, staging_buffer, nullptr);
        vkFreeMemory(device_, staging_buffer_memory, nullptr);
    }

    tuple<VkImageView, VkSampler> CreateTextureImage(void* data, VkDeviceSize image_size, uint2 size, cudaError_t& cuda_status) {

        CreateBuffer(image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);

        vkMapMemory(device_, staging_buffer_memory, 0, image_size, 0, &data);
        vkUnmapMemory(device_, staging_buffer_memory);

        //free(image_ptr);

        VkMemoryPropertyFlags alloc_flags = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        InitializeImage(image_, size, alloc_flags);
        InitializeImageViews();
        InitializeImageSampler();

        return tuple<VkImageView, VkSampler>(image_view_, image_sampler);
    }

    uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties mem_properties;
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties);
        for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
            if (typeFilter & (1 << i)) {
                return i;
            }
        }
        ProgramLog::OutputLine("Error: Failed to find suitable memory type!");
    }

    void AllocateImageMemory() {
        VkMemoryRequirements mem_requirements;
        vkGetImageMemoryRequirements(device_, image_, &mem_requirements);

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_requirements.size;
        alloc_info.memoryTypeIndex = FindMemoryType(mem_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(device_, &alloc_info, nullptr, &texture_image_memory) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed to allocate image memory!");
        }

        vkBindImageMemory(device_, image_, texture_image_memory, 0);
    }

    void InitializeImage(VkImage image, uint2 size, VkMemoryPropertyFlags properties) {
        VkImageCreateInfo image_info {};

        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.extent.width = static_cast<uint32_t>(size.x);
        image_info.extent.height = static_cast<uint32_t>(size.y);
        image_info.extent.depth = 1;
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;

        image_info.format = image_format_;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.flags = 0;

        if (vkCreateImage(device_, &image_info, nullptr, &image) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed to create image!");
        }

        VkMemoryRequirements mem_requirements;
        vkGetImageMemoryRequirements(device_, image_, &mem_requirements);

        VkMemoryAllocateInfo alloc_info = CreateAllocationInfo(mem_requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device_, &alloc_info, nullptr, &texture_image_memory) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed to allocate image memory!");
        }
        vkBindImageMemory(device_, image, texture_image_memory, 0);
    }

    VkMemoryAllocateInfo CreateAllocationInfo(VkMemoryRequirements& mem_requirements, VkMemoryPropertyFlags properties) {
        VkMemoryAllocateInfo alloc_info {};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_requirements.size;
        alloc_info.memoryTypeIndex = FindMemoryType(mem_requirements.memoryTypeBits, properties);

        return alloc_info;
    }

    void InitializeImageViews() {
        VkImageViewCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        create_info.image = image_;

        create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        create_info.format = image_format_;

        create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        create_info.subresourceRange.baseMipLevel = 0;
        create_info.subresourceRange.levelCount = 1;
        create_info.subresourceRange.baseArrayLayer = 0;
        create_info.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device_, &create_info, nullptr, &image_view_) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed to create image views!");
        }
    }

    void InitializeImageSampler() {
        VkSamplerCreateInfo sampler_info{};
        sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;

        if (vkCreateSampler(device_, &sampler_info, nullptr, &image_sampler) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed to create image sampler!");
        }
    }

    void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& buffer_memory) {
        VkBufferCreateInfo buffer_info {};

        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size;
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed to properly allocate buffer!");
        }

        VkMemoryRequirements mem_requirements;
        vkGetBufferMemoryRequirements(device_, buffer, &mem_requirements);

        VkMemoryAllocateInfo alloc_info = CreateAllocationInfo(mem_requirements, properties);

        if (vkAllocateMemory(device_, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device_, buffer, buffer_memory, 0);
    }

    VkResult StartRenderCommand() {
        VkCommandPoolCreateInfo pool_info {};

        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = queue_family_;

        if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }

        VkCommandBufferAllocateInfo alloc_info = AllocateBufferCommandInfo(command_pool);

        vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer);

        VkCommandBufferBeginInfo begin_info {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        return vkBeginCommandBuffer(command_buffer, &begin_info);
    }

    VkCommandBufferAllocateInfo AllocateBufferCommandInfo(VkCommandPool command_pool) {
        VkCommandBufferAllocateInfo alloc_info {};

        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        return alloc_info;
    }

    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;

    VkImage image_;
    VkImageView image_view_;
    VkSampler image_sampler;

    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;

    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory, texture_image_memory;

    VkFormat image_format_;

    uint32_t queue_family_;
};