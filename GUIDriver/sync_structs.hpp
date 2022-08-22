#pragma once

//Logging
#include "../CUDATest/handler_classes.hpp"

#include <vulkan/vulkan.h>

struct SyncStruct {
	SyncStruct() = default;

	SyncStruct(VkDevice& device_in) {
		device_ = device_in;
	}

	VkFenceCreateInfo FenceInfo(VkFenceCreateFlags flags) {
		VkFenceCreateInfo fence_info = {};

		fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fence_info.pNext = nullptr;
		fence_info.flags = flags;

		return fence_info;
	}

	VkSemaphoreCreateInfo SemaphoreInfo(VkSemaphoreCreateFlags flags) {
		VkSemaphoreCreateInfo semaphore_info = {};

		semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		semaphore_info.pNext = nullptr;
		semaphore_info.flags = flags;

		return semaphore_info;
	}

	VkResult CreateFence(VkFence& fence, VkFenceCreateInfo& fence_info) {
		vulkan_status = vkCreateFence(device_, &fence_info, nullptr, &fence);
		ProgramLog::OutputLine("\nSuccessfully created render fence!\n");

		return vulkan_status;
	}

	VkResult CreateSemaphore(VkSemaphore& semaphore, VkSemaphoreCreateInfo& semaphore_info) {
		vulkan_status = vkCreateSemaphore(device_, &semaphore_info, nullptr, &semaphore);
		ProgramLog::OutputLine("\nSuccessfully created render semaphore!\n");

		return vulkan_status;
	}

	void StartSyncStructs() {
		auto fence_info = FenceInfo(VK_FENCE_CREATE_SIGNALED_BIT);
		vulkan_status = CreateFence(fence_, fence_info);

		auto semaphore_info = SemaphoreInfo(0);
		vulkan_status = CreateSemaphore(present_semaphore_, semaphore_info);
		vulkan_status = CreateSemaphore(render_semaphore_, semaphore_info);
	}

	VkDevice device_;

	VkFence fence_;
	VkSemaphore render_semaphore_, present_semaphore_;

	VkResult vulkan_status;
};