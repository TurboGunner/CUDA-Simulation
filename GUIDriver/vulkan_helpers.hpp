#pragma once

#include "gui_driver.cuh"
#include "image_helpers.hpp"

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

	static VkCommandBufferBeginInfo BeginCommandBufferInfo() {
		VkCommandBufferBeginInfo begin_info = {};

		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		return begin_info;
	}

	void CreateSwapchainFrameBuffers(vector<VkImageView>& swapchain_image_views) {
		frame_buffers_.resize(swapchain_image_views.size());

		VkFramebufferCreateInfo frame_buffer_info = {};
		frame_buffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		frame_buffer_info.pNext = nullptr;

		frame_buffer_info.renderPass = render_pass_;
		frame_buffer_info.attachmentCount = 1;
		frame_buffer_info.width = size_.x;
		frame_buffer_info.height = size_.y;
		frame_buffer_info.layers = 1;

		frame_buffers_ = vector<VkFramebuffer>(swapchain_image_views.size());

		for (size_t i = 0; i < swapchain_image_views.size(); i++) {
			frame_buffer_info.pAttachments = &swapchain_image_views[i];
			vulkan_status = vkCreateFramebuffer(device_, &frame_buffer_info, nullptr, &frame_buffers_[i]);
		}
	}

	VkCommandPool CreateCommandPool(VkCommandPool& command_pool, const uint32_t& queue_family) {
		VkCommandPoolCreateInfo pool_info = {};

		pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		pool_info.queueFamilyIndex = queue_family;

		if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool!");
		}

		return command_pool;
	}

	VkDevice device_;

	VkRenderPass render_pass_;

	vector<VkFramebuffer> frame_buffers_;

	VkFormat swapchain_format_;
	VkExtent2D swapchain_extent_;

	uint2 size_;

	VkResult vulkan_status;
};