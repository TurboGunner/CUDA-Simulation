#include "sync_structs.hpp"

SyncStruct::SyncStruct(VkDevice& device_in, const size_t& max_frames_const_in) {
	device_ = device_in;
	MAX_FRAMES_IN_FLIGHT_ = max_frames_const_in;
}

VkResult SyncStruct::Initialize() {
	VkResult vulkan_status = VK_SUCCESS;
	auto fence_info = FenceInfo(VK_FENCE_CREATE_SIGNALED_BIT);
	vulkan_status = CreateFences(fences_, fence_info, MAX_FRAMES_IN_FLIGHT_);

	auto semaphore_info = SemaphoreInfo();
	vulkan_status = CreateSemaphores(render_semaphores_, semaphore_info, MAX_FRAMES_IN_FLIGHT_);
	vulkan_status = CreateSemaphores(present_semaphores_, semaphore_info, MAX_FRAMES_IN_FLIGHT_);

	return vulkan_status;
}

void SyncStruct::Clean() {
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT_; i++) {
		vkDestroySemaphore(device_, render_semaphores_[i], nullptr);
		vkDestroySemaphore(device_, present_semaphores_[i], nullptr);
		vkDestroyFence(device_, fences_[i], nullptr);
	}
}

void SyncStruct::GetWaitSemaphores(const size_t& current_frame) {
	if (current_frame != 0) {
		wait_semaphores_.push_back(vk_wait_semaphore_);
		wait_stages_.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	}
}

void SyncStruct::GetSignalFrameSemaphores() {
	signal_semaphores_.push_back(vk_signal_semaphore_);
}

void SyncStruct::ClearInteropSynchronization() {
	wait_semaphores_.clear();
	signal_semaphores_.clear();

	wait_stages_.clear();
}

VkFenceCreateInfo SyncStruct::FenceInfo(const VkFenceCreateFlags& flags) {
	VkFenceCreateInfo fence_info = {};

	fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fence_info.pNext = nullptr;
	fence_info.flags = flags;

	return fence_info;
}

VkSemaphoreCreateInfo SyncStruct::SemaphoreInfo(const VkSemaphoreCreateFlags& flags) {
	VkSemaphoreCreateInfo semaphore_info = {};

	semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	semaphore_info.pNext = nullptr;
	semaphore_info.flags = flags;

	return semaphore_info;
}

VkResult SyncStruct::CreateFence(VkFence& fence, const VkFenceCreateInfo& fence_info, const bool& log) { //False due to not using individual fence inits anymore
	VkResult vulkan_status = VK_SUCCESS;
	vulkan_status = vkCreateFence(device_, &fence_info, nullptr, &fence);

	if (log) {
		ProgramLog::OutputLine("Successfully created render fence!\n");
	}

	return vulkan_status;
}


VkResult SyncStruct::CreateFences(vector<VkFence>& fences, const VkFenceCreateInfo& fence_info, const size_t& size) {
	VkResult vulkan_status = VK_SUCCESS;
	fences.resize(size);

	for (size_t i = 0; i < size && vulkan_status == VK_SUCCESS; i++) {
		vulkan_status = CreateFence(fences[i], fence_info);
		ProgramLog::OutputLine("Successfully created render fence #" + std::to_string(i) + "!");
	}

	ProgramLog::OutputLine("Successfully created all render fences!\n");

	return vulkan_status;
}

VkResult SyncStruct::CreateSemaphore(VkSemaphore& semaphore, const VkSemaphoreCreateInfo& semaphore_info, const bool& log) {
	VkResult vulkan_status = VK_SUCCESS;
	vulkan_status = vkCreateSemaphore(device_, &semaphore_info, nullptr, &semaphore);

	if (log) {
		ProgramLog::OutputLine("Successfully created render semaphore!\n");
	}

	return vulkan_status;
}

VkResult SyncStruct::CreateSemaphores(vector<VkSemaphore>& semaphores, const VkSemaphoreCreateInfo& semaphore_info, const size_t& size) {
	VkResult vulkan_status = VK_SUCCESS;
	semaphores.resize(size);

	for (size_t i = 0; i < size && vulkan_status == VK_SUCCESS; i++) {
		vulkan_status = CreateSemaphore(semaphores[i], semaphore_info);
		ProgramLog::OutputLine("Successfully created render semaphore #" + std::to_string(i) + "!");
	}

	ProgramLog::OutputLine("Successfully created all render semaphores!\n");

	return vulkan_status;
}