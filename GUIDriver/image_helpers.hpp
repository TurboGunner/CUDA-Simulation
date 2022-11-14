#pragma once

#include "../CUDATest/handler_classes.hpp"

#include "vulkan_helpers.hpp"

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

struct ImageHelper {
	static void InitializeImage(VkDevice& device, VkPhysicalDevice& physical_device, VkDeviceMemory& texture_image_memory, VkImage& image, uint2& size,
		const VkFormat image_format, const VkImageUsageFlags usage_flags, const VkMemoryPropertyFlags properties, const VkImageType image_type = VK_IMAGE_TYPE_2D);

	static VkImageView CreateImageView(VkDevice& device, VkImage& image, const VkFormat format, const VkImageAspectFlags flags, const VkImageViewType type = VK_IMAGE_VIEW_TYPE_2D);

	static VkFormat FindDepthFormat(VkPhysicalDevice& physical_device);

	static VkFormat FindSupportedFormat(VkPhysicalDevice& physical_device, const vector<VkFormat>& candidates, const VkImageTiling tiling, const VkFormatFeatureFlags features);

	static void TransitionImageLayout(VkDevice& device, VkCommandPool& command_pool, VkQueue& queue, VkImage image, const VkFormat format, const VkImageLayout old_layout, const VkImageLayout new_layout);

private:
	static VkImageMemoryBarrier CreateImageMemoryBarrier(VkImage image, const VkImageLayout old_layout, const VkImageLayout new_layout);
};