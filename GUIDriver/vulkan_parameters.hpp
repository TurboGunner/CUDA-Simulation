#pragma once

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

#include <deque>
#include <functional>
#include <string>

using std::deque;
using std::function;
using std::string;

class VulkanParameters {
public:
	VulkanParameters() = default;

	VkDevice device_;
	VkPhysicalDevice physical_device_;

	VkSurfaceKHR surface_;
	VkSurfaceFormatKHR surface_format_;

	VkSwapchainKHR swapchain_;

	VkRenderPass render_pass_;

	VkCommandPool command_pool_;

	VkPipeline pipeline_;
	VkPipelineCache pipeline_cache_;

	VkQueue queue_ = VK_NULL_HANDLE;

	VkSemaphore image_semaphore_, render_semaphore_;
	VkFence render_fence_;

	VkViewport viewport_;
	VkRect2D scissor_;

	uint32_t image_index_, current_frame_, frame_index_;

	uint2 size_;
};

struct AllocationParams {
	AllocationParams() = default;

	AllocationParams(function<void()> func_in, string name_in = "") {
		name_ = name_in;
		delete_func_ = func_in;
	}

	function<void()> delete_func_;
	string name_;
};

struct VulkanMemoryManager {

	void PushFunction(function<void()>& function, string name = "") {
		allocations_.push_back(AllocationParams(function, name));
	}

	void Flush() {
		for (auto it = allocations_.rbegin(); it != allocations_.rend(); it++) {
			(*it).delete_func_();
		}

		allocations_.clear();
	}
private:
	deque<AllocationParams> allocations_;
};