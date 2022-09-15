#pragma once

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

#include <deque>
#include <functional>
#include <string>
#include <vector>

using std::deque;
using std::function;
using std::string;
using std::vector;

class VulkanParameters {
public:
	VulkanParameters() = default;

	VkDevice device_;
	VkPhysicalDevice physical_device_;

	VkSurfaceKHR surface_;
	VkSurfaceFormatKHR surface_format_;

	VkSwapchainKHR swapchain_;

	VkRenderPass render_pass_;

	vector<VkCommandBuffer> command_buffers_;

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