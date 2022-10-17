#pragma once

#undef CreateSemaphore

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

//Logging
#include "../CUDATest/handler_classes.hpp"

#include <vulkan/vulkan.h>

#include <vector>

using std::vector;

class SyncStruct {
public:
	SyncStruct() = default;

	SyncStruct(VkDevice& device_in, const size_t& max_frames_const_in);

	VkResult Initialize();

	void Clean();

	void CleanSynchronization();

	void GetWaitSemaphores(const size_t& current_frame);

	void GetSignalFrameSemaphores();

	void ClearInteropSynchronization();

	vector<VkFence> fences_;
	vector<VkSemaphore> render_semaphores_, present_semaphores_; //Wait, Signal

	vector<VkSemaphore> wait_semaphores_, signal_semaphores_;

	VkSemaphore vk_wait_semaphore_ = {}, vk_signal_semaphore_ = {};

	vector<VkPipelineStageFlags> wait_stages_;


private:
	VkFenceCreateInfo FenceInfo(const VkFenceCreateFlags& flags = 0);

	VkSemaphoreCreateInfo SemaphoreInfo(const VkSemaphoreCreateFlags& flags = 0);

	VkResult CreateFence(VkFence& fence, const VkFenceCreateInfo& fence_info, const bool& log = false);


	VkResult CreateFences(vector<VkFence>& fences, const VkFenceCreateInfo& fence_info, const size_t& size);

	VkResult CreateSemaphore(VkSemaphore& semaphore, const VkSemaphoreCreateInfo& semaphore_info, const bool& log = false);

	VkResult CreateSemaphores(vector<VkSemaphore>& semaphores, const VkSemaphoreCreateInfo& semaphore_info, const size_t& size);

	VkDevice device_;

	size_t MAX_FRAMES_IN_FLIGHT_;
};