#include "image_helpers.hpp"

void ImageHelper::InitializeImage(VkDevice& device, VkPhysicalDevice& physical_device, VkDeviceMemory& texture_image_memory, VkImage& image, uint2& size,
	const VkFormat& image_format, const VkImageUsageFlags& usage_flags, const VkMemoryPropertyFlags& properties, const VkImageType& image_type) {
	VkImageCreateInfo image_info = {};

	image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_info.imageType = image_type;
	image_info.extent.width = size.x;
	image_info.extent.height = size.y;
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
	vkGetImageMemoryRequirements(device, image, &mem_requirements);

	VkMemoryAllocateInfo alloc_info = VulkanHelper::CreateAllocationInfo(physical_device, mem_requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	if (vkAllocateMemory(device, &alloc_info, nullptr, &texture_image_memory) != VK_SUCCESS) {
		ProgramLog::OutputLine("Error: Failed to allocate image memory!");
	}
	vkBindImageMemory(device, image, texture_image_memory, 0);
}

VkImageView ImageHelper::CreateImageView(VkDevice& device, VkImage& image, const VkFormat& format, const VkImageAspectFlags& flags, const VkImageViewType& type) {
	VkImageView image_view = {};

	VkImageViewCreateInfo create_info = {};

	create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	create_info.image = image;

	create_info.viewType = type;
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

VkFormat ImageHelper::FindDepthFormat(VkPhysicalDevice& physical_device) {
	return FindSupportedFormat(physical_device, { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
		VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

VkFormat ImageHelper::FindSupportedFormat(VkPhysicalDevice& physical_device, const vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
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

void ImageHelper::TransitionImageLayout(VkDevice& device, VkCommandPool& command_pool, VkQueue& queue, VkImage image, const VkFormat& format, const VkImageLayout& old_layout, const VkImageLayout& new_layout) {
	VkCommandBuffer command_buffer = VulkanHelper::BeginSingleTimeCommands(device, command_pool);

	VkImageMemoryBarrier barrier = CreateImageMemoryBarrier(image, old_layout, new_layout);

	VkPipelineStageFlags source_stage = {};
	VkPipelineStageFlags destination_stage = {};

	if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

		if (format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT) {
			barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
		}
	}
	else {
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	}

	if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}

	else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destination_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	}

	else {
		ProgramLog::OutputLine("\nSuccessfully submitted item to queue!\n");
	}

	vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
	VulkanHelper::EndSingleTimeCommands(command_buffer, device, command_pool, queue);
}

VkImageMemoryBarrier ImageHelper::CreateImageMemoryBarrier(VkImage image, const VkImageLayout& old_layout, const VkImageLayout& new_layout) {
	VkImageMemoryBarrier barrier = {};

	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;

	barrier.oldLayout = old_layout;
	barrier.newLayout = new_layout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;

	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;

	return barrier;
}