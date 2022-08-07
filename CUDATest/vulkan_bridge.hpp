#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "../Renderer/raypath.cuh"

#include <vulkan/vulkan.h>
#include <imgui_impl_vulkan.h>

#include <vector>

using std::vector;

class VulkanImageDisplay {
	void CreateTextureImage(uint2 size, cudaError_t& cuda_status) {

		VkDeviceSize imageSize = size.x * size.y * 4;

		Vector3D* image_ptr = AllocateTexture(size, cuda_status);
		vector<Vector3D> image = OutputImage(image_ptr, size);

	}

	void InitializeImage(uint2 size) {
		VkImageCreateInfo imageInfo{};

		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = static_cast<uint32_t>(size.x);
		imageInfo.extent.height = static_cast<uint32_t>(size.y);
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;

		imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.flags = 0;

		if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image views!");
		}
	}
};