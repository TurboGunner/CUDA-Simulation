#pragma once

#include "gui_driver.cuh"

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

	VkCommandBufferAllocateInfo AllocateCommandBuffer(VkCommandPool pool, uint32_t count = 1, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
		VkCommandBufferAllocateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		info.pNext = nullptr;

		info.commandPool = pool;
		info.commandBufferCount = count;
		info.level = level;
		return info;
	}

	void CreateFrameBuffer() {
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
			//VulkanErrorHandler(vulkan_status);
		}
	}

	VkDevice device_;

	VkRenderPass render_pass_;

	vector<VkFramebuffer> frame_buffers_;
	vector<VkImageView> swapchain_image_views_;

	uint2 size_;

	VkResult vulkan_status;
};