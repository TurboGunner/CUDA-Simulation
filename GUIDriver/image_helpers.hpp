#pragma once

#include "../CUDATest/handler_classes.hpp"

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

struct ImageHelper {
	static uint32_t FindMemoryType(VkPhysicalDevice& physical_device, const uint32_t& type_filter, const VkMemoryPropertyFlags& properties) {
		VkPhysicalDeviceMemoryProperties mem_properties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);
		for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
			if (type_filter & (1 << i)) {
				return i;
			}
		}
		ProgramLog::OutputLine("Error: Failed to find suitable memory type!");
	}

	static VkMemoryAllocateInfo CreateAllocationInfo(VkPhysicalDevice& physical_device, VkMemoryRequirements& mem_requirements, VkMemoryPropertyFlags properties) {
		VkMemoryAllocateInfo alloc_info{};
		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = mem_requirements.size;
		alloc_info.memoryTypeIndex = FindMemoryType(physical_device, mem_requirements.memoryTypeBits, properties);

		return alloc_info;
	}

	static void InitializeImage(VkDevice& device, VkPhysicalDevice& physical_device, VkDeviceMemory& texture_image_memory, VkImage& image, uint2& size,
		const VkFormat& image_format, const VkImageUsageFlags usage_flags, const VkMemoryPropertyFlags& properties) {
		VkImageCreateInfo image_info = {};

		image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		image_info.imageType = VK_IMAGE_TYPE_2D;
		image_info.extent.width = static_cast<uint32_t>(size.x);
		image_info.extent.height = static_cast<uint32_t>(size.y);
		image_info.extent.depth = 1;
		image_info.mipLevels = 1;
		image_info.arrayLayers = 1;

		image_info.format = image_format;
		image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
		image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		image_info.usage = usage_flags;
		image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		image_info.samples = VK_SAMPLE_COUNT_1_BIT;
		image_info.flags = 0;

		if (vkCreateImage(device, &image_info, nullptr, &image) != VK_SUCCESS) {
			ProgramLog::OutputLine("Error: Failed to create image!");
		}

		VkMemoryRequirements mem_requirements = {}; //NOTE
		mem_requirements.size = size.x * size.y * 4;

		VkMemoryAllocateInfo alloc_info = CreateAllocationInfo(physical_device, mem_requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		if (vkAllocateMemory(device, &alloc_info, nullptr, &texture_image_memory) != VK_SUCCESS) {
			ProgramLog::OutputLine("Error: Failed to allocate image memory!");
		}
		vkBindImageMemory(device, image, texture_image_memory, 0);
	}

	static VkImageView CreateImageView(VkDevice& device, VkImage& image, const VkFormat& format, const VkImageAspectFlags& flags) {
		VkImageView image_view = {};

		VkImageViewCreateInfo create_info = {};

		create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		create_info.image = image;

		create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
		create_info.format = format;

		create_info.subresourceRange.aspectMask = flags;
		create_info.subresourceRange.baseMipLevel = 0;
		create_info.subresourceRange.levelCount = 1;
		create_info.subresourceRange.baseArrayLayer = 0;
		create_info.subresourceRange.layerCount = 1;

		if (vkCreateImageView(device, &create_info, nullptr, &image_view) != VK_SUCCESS) {
			ProgramLog::OutputLine("Error: Failed to create image views!");
		}
		return image_view;
	}

	static VkFormat FindDepthFormat(VkPhysicalDevice& physical_device) {
		return FindSupportedFormat(physical_device, { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
	}

	static VkFormat FindSupportedFormat(VkPhysicalDevice& physical_device, const vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
		for (VkFormat format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);

			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
				return format;
			}
		}
	}

};