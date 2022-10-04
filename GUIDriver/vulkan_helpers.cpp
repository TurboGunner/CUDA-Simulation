#include "vulkan_helpers.hpp"

VulkanHelper::VulkanHelper(VkDevice& device_in, uint2& size_in, const size_t& max_frames_const_in) {
	device_ = device_in;

	size_ = size_in;

	MAX_FRAMES_IN_FLIGHT_ = max_frames_const_in;
}

VkCommandBufferAllocateInfo VulkanHelper::AllocateCommandBufferInfo(VkCommandPool& pool, uint32_t count, VkCommandBufferLevel level) {
	VkCommandBufferAllocateInfo info = {};

	info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	info.pNext = nullptr;

	info.commandPool = pool;
	info.commandBufferCount = count;
	info.level = level;

	return info;
}

VkResult VulkanHelper::InitializeCommandBuffer(VkDevice& device, VkCommandBuffer& command_buffer, VkCommandPool& command_pool) {
	auto allocation_info = AllocateCommandBufferInfo(command_pool);

	return vkAllocateCommandBuffers(device, &allocation_info, &command_buffer);
}

VkResult VulkanHelper::InitializeCommandBuffers(vector<VkCommandBuffer>& command_buffers, VkCommandPool& command_pool) {
	command_buffers.resize(MAX_FRAMES_IN_FLIGHT_);

	auto allocation_info = AllocateCommandBufferInfo(command_pool, MAX_FRAMES_IN_FLIGHT_);

	return vkAllocateCommandBuffers(device_, &allocation_info, command_buffers.data());
}

VkCommandBufferBeginInfo VulkanHelper::BeginCommandBufferInfo() {
	VkCommandBufferBeginInfo begin_info = {};

	begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	return begin_info;
}

VkCommandPool VulkanHelper::CreateCommandPool(VkCommandPool& command_pool, const uint32_t& queue_family) {
	VkCommandPoolCreateInfo pool_info = {};

	pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	pool_info.queueFamilyIndex = queue_family;

	if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool) != VK_SUCCESS) {
		throw std::runtime_error("Error: Failed to create command pool!");
	}

	return command_pool;
}

uint32_t VulkanHelper::FindMemoryType(VkPhysicalDevice& physical_device, uint32_t type_filter, const VkMemoryPropertyFlags& properties, const bool& log) {
	VkPhysicalDeviceMemoryProperties mem_properties;
	vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

	for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
		if (type_filter & (1 << i) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
			if (log) {
				ProgramLog::OutputLine("Memory Type: " + std::to_string(i));
			}
			return i;
		}
	}
	ProgramLog::OutputLine("Error: Failed to find suitable memory type!");
}

VkMemoryAllocateInfo VulkanHelper::CreateAllocationInfo(VkPhysicalDevice& physical_device, VkMemoryRequirements& mem_requirements, const VkMemoryPropertyFlags& properties, bool log) {
	VkMemoryAllocateInfo alloc_info = {};

	alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	alloc_info.allocationSize = mem_requirements.size;
	alloc_info.memoryTypeIndex = FindMemoryType(physical_device, mem_requirements.memoryTypeBits, properties, log);

	if (log) {
		ProgramLog::OutputLine("Created allocation info for video memory.");
	}

	return alloc_info;
}

VkCommandBuffer VulkanHelper::BeginSingleTimeCommands(VkDevice& device, VkCommandPool command_pool, bool log) {

	VkCommandBuffer command_buffer;
	InitializeCommandBuffer(device, command_buffer, command_pool);

	auto begin_info = BeginCommandBufferInfo();

	vkBeginCommandBuffer(command_buffer, &begin_info);

	if (log) {
		ProgramLog::OutputLine("Started command recording!");
	}

	return command_buffer;
}


VkResult VulkanHelper::EndSingleTimeCommands(VkCommandBuffer& command_buffer, VkDevice& device, VkCommandPool command_pool, VkQueue queue, const bool& log, const size_t& size) {
	VkResult vulkan_status = VK_SUCCESS;
	vulkan_status = vkEndCommandBuffer(command_buffer);

	if (log) {
		ProgramLog::OutputLine("Ended command recording!");
	}

	VkSubmitInfo submit_info = {};

	submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount = size;
	submit_info.pCommandBuffers = &command_buffer;

	vulkan_status = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);

	if (log) {
		ProgramLog::OutputLine("\nSuccessfully submitted item to queue!\n");
	}
	vulkan_status = vkQueueWaitIdle(queue);

	vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);

	return vulkan_status;
}