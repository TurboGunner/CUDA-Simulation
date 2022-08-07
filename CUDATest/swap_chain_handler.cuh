#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <vulkan/vulkan.h>

//Logging
#include "handler_classes.hpp"

//Renderer
#include "../Renderer/raypath.cuh"

#include <functional>
#include <tuple>
#include <vector>

using std::tuple;
using std::vector;
using std::reference_wrapper;

class SwapChainHandler {
public:
    SwapChainHandler(VkDevice& device_in, VkPhysicalDevice& physical_in) {
        device_.push_back(reference_wrapper<VkDevice>(device_in));
        physical_device_ = physical_in;
    }

    ~SwapChainHandler() {
        vkDestroyImageView(device_[0].get(), swap_chain_image_view, nullptr);
    }

    tuple<VkImageView, VkSampler> CreateTextureImage(uint2 size, cudaError_t& cuda_status) {

        VkDeviceSize image_size = size.x * size.y * sizeof(Vector3D);

        void* data = (void*)AllocateTexture(size, cuda_status);

        CreateBuffer(image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);

        vkMapMemory(device_[0].get(), staging_buffer_memory, 0, image_size, 0, &data);
        vkUnmapMemory(device_[0].get(), staging_buffer_memory);

        //free(image_ptr);

        VkMemoryPropertyFlags alloc_flags = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        InitializeImage(size, alloc_flags);
        InitializeImageViews();
        InitializeImageSampler();

        return tuple<VkImageView, VkSampler>(swap_chain_image_view, image_sampler);
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
        vkGetImageMemoryRequirements(device_[0].get(), swap_chain_image, &mem_requirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = mem_requirements.size;
        allocInfo.memoryTypeIndex = FindMemoryType(mem_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(device_[0].get(), &allocInfo, nullptr, &texture_image_memory) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed to allocate image memory!");
        }

        vkBindImageMemory(device_[0].get(), swap_chain_image, texture_image_memory, 0);
    }

    void InitializeImage(uint2 size, VkMemoryPropertyFlags properties) {
        VkImageCreateInfo image_info {};

        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.extent.width = static_cast<uint32_t>(size.x);
        image_info.extent.height = static_cast<uint32_t>(size.y);
        image_info.extent.depth = 1;
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;

        image_info.format = VK_FORMAT_R8G8B8A8_SRGB;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.flags = 0;

        if (vkCreateImage(device_[0].get(), &image_info, nullptr, &swap_chain_image) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed to create image!");
        }

        VkMemoryRequirements mem_requirements;
        vkGetImageMemoryRequirements(device_[0], swap_chain_image, &mem_requirements);

        VkMemoryAllocateInfo alloc_info = CreateAllocationInfo(mem_requirements, properties);
        if (vkAllocateMemory(device_[0], &alloc_info, nullptr, &texture_image_memory) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed to allocate image memory!");
        }
    }

    VkMemoryAllocateInfo CreateAllocationInfo(VkMemoryRequirements mem_requirements, VkMemoryPropertyFlags properties) {
        VkMemoryAllocateInfo alloc_info {};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_requirements.size;
        alloc_info.memoryTypeIndex = FindMemoryType(mem_requirements.memoryTypeBits, properties);

        return alloc_info;
    }

    void InitializeImageViews() {
        VkImageViewCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        create_info.image = swap_chain_image;

        create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        create_info.format = swap_chain_image_format;

        create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        create_info.subresourceRange.baseMipLevel = 0;
        create_info.subresourceRange.levelCount = 1;
        create_info.subresourceRange.baseArrayLayer = 0;
        create_info.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device_[0].get(), &create_info, nullptr, &swap_chain_image_view) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed to create image views!");
        }
    }

    void InitializeImageSampler() {
        VkSamplerCreateInfo sampler_info{};
        sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;

        if (vkCreateSampler(device_[0].get(), &sampler_info, nullptr, &image_sampler) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed to create image sampler!");
        }
    }

    void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device_[0].get(), &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            ProgramLog::OutputLine("Error: Failed to properly allocate buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device_[0].get(), buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device_[0].get(), &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device_[0].get(), buffer, bufferMemory, 0);
    }

    vector<reference_wrapper<VkDevice>> device_;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;

    VkImage swap_chain_image;
    VkImageView swap_chain_image_view;
    VkSampler image_sampler;

    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory, texture_image_memory;

    VkSwapchainKHR swap_chain;
    VkFormat swap_chain_image_format;
    VkExtent2D swap_chain_extent;
};