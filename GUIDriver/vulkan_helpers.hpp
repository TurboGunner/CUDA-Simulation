#pragma once

#include "gui_driver.cuh"

//Logging
#include "../CUDATest/handler_classes.hpp"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <vulkan/vulkan.h>

#include <vector>

using std::vector;

struct VulkanHelper {
	VulkanHelper() = default;

	VulkanHelper(VkDevice& device_in, VkRenderPass& pass_in) {
		device_ = device_in;

		render_pass_ = pass_in;
	}

	static VkCommandBufferAllocateInfo AllocateCommandBuffer(VkCommandPool& pool, uint32_t count = 1, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
		VkCommandBufferAllocateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		info.pNext = nullptr;

		info.commandPool = pool;
		info.commandBufferCount = count;
		info.level = level;
		return info;
	}

	void CreateFrameBuffers() {
		VkFramebufferCreateInfo frame_buffer_info = {};
		frame_buffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		frame_buffer_info.pNext = nullptr;

		frame_buffer_info.renderPass = render_pass_;
		frame_buffer_info.attachmentCount = 1;
		frame_buffer_info.width = size_.x;
		frame_buffer_info.height = size_.y;
		frame_buffer_info.layers = 1;

		frame_buffers_ = vector<VkFramebuffer>(swapchain_image_views_.size());

		for (size_t i = 0; i < swapchain_image_views_.size(); i++) {
			frame_buffer_info.pAttachments = &swapchain_image_views_[i];
			vulkan_status = vkCreateFramebuffer(device_, &frame_buffer_info, nullptr, &frame_buffers_[i]);
		}
	}


	VkImageView CreateImageView(VkImage image, VkFormat format, VkImageAspectFlags flags) {
		VkImageView image_view;

		VkImageViewCreateInfo create_info {};

		create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		create_info.image = image;

		create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
		create_info.format = format;

		create_info.subresourceRange.aspectMask = flags;
		create_info.subresourceRange.baseMipLevel = 0;
		create_info.subresourceRange.levelCount = 1;
		create_info.subresourceRange.baseArrayLayer = 0;
		create_info.subresourceRange.layerCount = 1;

		if (vkCreateImageView(device_, &create_info, nullptr, &image_view) != VK_SUCCESS) {
			ProgramLog::OutputLine("Error: Failed to create image views!");
		}
		return image_view;
	}

	void CreateImageViews() {
		swapchain_image_views_.resize(swapchain_image_views_.size());

		for (uint32_t i = 0; i < swapchain_image_views_.size(); i++) {
			swapchain_image_views_[i] = CreateImageView(swapchain_images_[i], swapchain_format_, VK_IMAGE_ASPECT_COLOR_BIT);
		}
	}

	VkRenderPassBeginInfo SubpassBeginInfo(VkRenderPass& subpass, VkFramebuffer& frame_buffer) {
		VkRenderPassBeginInfo pass_info = {};

		pass_info.renderPass = subpass;
		pass_info.renderArea.offset.x = 0;
		pass_info.renderArea.offset.y = 0;
		pass_info.renderArea.extent.width = size_.x;
		pass_info.renderArea.extent.height = size_.y;
		pass_info.clearValueCount = 1;
		pass_info.pClearValues = nullptr;
		pass_info.framebuffer = frame_buffer;
;
		return pass_info;
	}

	VkCommandPool CreateCommandPool(VkCommandPool& command_pool, const uint32_t& queue_family) {
		VkCommandPoolCreateInfo pool_info {};
		pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		pool_info.queueFamilyIndex = queue_family;

		if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool!");
		}
	}

	VkDevice device_;

	VkRenderPass render_pass_;

	vector<VkFramebuffer> frame_buffers_;

	vector<VkImage> swapchain_images_;
	vector<VkImageView> swapchain_image_views_;
	VkFormat swapchain_format_;
	VkExtent2D swapchain_extent_;

	uint2 size_;

	VkResult vulkan_status;
};