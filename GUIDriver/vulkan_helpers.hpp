#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

//Logging
#include "../CUDATest/handler_classes.hpp"

#include <vulkan/vulkan.h>

#include <vector>

using std::vector;

struct VulkanHelper {
	VulkanHelper() = default;

	VulkanHelper(VkDevice& device_in, uint2& size_in, const size_t& max_frames_const_in);

	static VkCommandBufferAllocateInfo AllocateCommandBufferInfo(VkCommandPool& pool, uint32_t count = 1, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

	static VkResult InitializeCommandBuffer(VkDevice& device, VkCommandBuffer& command_buffer, VkCommandPool& command_pool);

	VkResult InitializeCommandBuffers(vector<VkCommandBuffer>& command_buffers, VkCommandPool& command_pool);

	static VkCommandBufferBeginInfo BeginCommandBufferInfo();

	VkCommandPool CreateCommandPool(VkCommandPool& command_pool, const uint32_t& queue_family);

	static uint32_t FindMemoryType(VkPhysicalDevice& physical_device, uint32_t type_filter, const VkMemoryPropertyFlags& properties, const bool& log = true);

	static VkMemoryAllocateInfo CreateAllocationInfo(VkPhysicalDevice& physical_device, VkMemoryRequirements& mem_requirements, const VkMemoryPropertyFlags& properties, bool log = true);

	static VkCommandBuffer BeginSingleTimeCommands(VkDevice& device, VkCommandPool command_pool, bool log = true);

	static VkResult EndSingleTimeCommands(VkCommandBuffer& command_buffer, VkDevice& device, VkCommandPool command_pool, VkQueue queue, const bool& log = true, const size_t& size = 1);

	VkDevice device_;

	size_t MAX_FRAMES_IN_FLIGHT_;

	uint2 size_;
};