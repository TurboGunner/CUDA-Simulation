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

	VulkanHelper(VkDevice& device_in, VkRenderPass& pass_in, uint2& size_in) {
		device_ = device_in;

		render_pass_ = pass_in; //NOTE: Probably no longer needed

		size_ = size_in;
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

	static uint32_t FindMemoryType(VkPhysicalDevice& physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties mem_properties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);
		for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
			if (type_filter & (1 << i) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
				ProgramLog::OutputLine("Memory Type: " + std::to_string(i));
				return i;
			}
		}
		ProgramLog::OutputLine("Error: Failed to find suitable memory type!");
	}

	static VkMemoryAllocateInfo CreateAllocationInfo(VkPhysicalDevice& physical_device, VkMemoryRequirements& mem_requirements, VkMemoryPropertyFlags properties) {
		VkMemoryAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = mem_requirements.size;
		alloc_info.memoryTypeIndex = FindMemoryType(physical_device, mem_requirements.memoryTypeBits, properties);

		ProgramLog::OutputLine("Created allocation info for video memory.");

		return alloc_info;
	}

	static VkCommandBuffer BeginSingleTimeCommands(VkDevice& device, VkCommandPool command_pool) {
		VkCommandBufferAllocateInfo alloc_info = AllocateCommandBuffer(command_pool);

		VkCommandBuffer command_buffer;
		vkAllocateCommandBuffers(device, &alloc_info, &command_buffer);

		auto begin_info = BeginCommandBufferInfo();

		vkBeginCommandBuffer(command_buffer, &begin_info);

		ProgramLog::OutputLine("Started command recording!");

		return command_buffer;
	}


	static void EndSingleTimeCommands(VkCommandBuffer& command_buffer, VkDevice& device, VkCommandPool command_pool, VkQueue queue) {
		vkEndCommandBuffer(command_buffer);

		ProgramLog::OutputLine("Ended command recording!");

		VkSubmitInfo submit_info = {};

		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;

		vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
		ProgramLog::OutputLine("\nSuccessfully submitted item to queue!\n");
		vkQueueWaitIdle(queue);

		vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
	}

	VkDevice device_;

	VkRenderPass render_pass_;

	uint2 size_;

	VkResult vulkan_status;
};