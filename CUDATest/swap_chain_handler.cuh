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
    SwapChainHandler(VkDevice& device_in) {
        device_.push_back(reference_wrapper<VkDevice>(device_in));
    }

    ~SwapChainHandler() {
        vkDestroyImageView(device_[0].get(), swap_chain_image_view, nullptr);
    }

    tuple<VkImageView, VkSampler> CreateTextureImage(uint2 size, cudaError_t& cuda_status) {

        VkDeviceSize imageSize = size.x * size.y * 4;

        Vector3D* image_ptr = AllocateTexture(size, cuda_status);
        vector<Vector3D> image = OutputImage(image_ptr, size);

        InitializeImage(size);
        InitializeImageViews();
        InitializeImageSampler();

        return tuple<VkImageView, VkSampler>(swap_chain_image_view, image_sampler);
    }

    void InitializeImage(uint2 size) {
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

    vector<reference_wrapper<VkDevice>> device_;

    VkImage swap_chain_image;
    VkImageView swap_chain_image_view;
    VkSampler image_sampler;

    VkSwapchainKHR swap_chain;
    VkFormat swap_chain_image_format;
    VkExtent2D swap_chain_extent;
};