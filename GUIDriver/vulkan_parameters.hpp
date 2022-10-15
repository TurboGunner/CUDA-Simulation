#pragma once

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

#include "cuda_interop_helper.cuh"
#include "rasterizer.cuh"
#include "shader_loader.cuh"
#include "swap_chain_manager.cuh"

#include "mesh_manager.hpp"
#include "sync_structs.hpp"
#include "vertex_data.hpp"
#include "mesh_viewport.hpp"
#include "descriptor_set_handler.hpp"

#include "vulkan_helpers.hpp"
#include "image_helpers.hpp"

#include "../Meshing/mpm.cuh"

#include <string>
#include <vector>

using std::string;
using std::vector;

struct InFlightObjects {
	VkSemaphore image_semaphore;
	VkSemaphore render_semaphore;
	VkFence fence;
	VkCommandBuffer command_buffer;
};

class VulkanParameters {
public:
	VulkanParameters() = default;

	VulkanParameters(VkDevice& device_in, VkPhysicalDevice& phys_device_in, VkSurfaceKHR& surface_in, const uint32_t& max_frames_in_flight_in, VkQueue& queue_in, const uint32_t& queue_family_in, VkDescriptorPool& descriptor_pool_in, uint2& size_in);

	VkResult InitVulkan();

	VkResult InitVulkanStage2();

	VkSemaphore& InFlightImageSemaphore();

	VkSemaphore& InFlightRenderSemaphore();

	VkFence& InFlightFence();

	VkCommandBuffer& InFlightCommandBuffer();

	InFlightObjects InFlightAll();

	VkFramebuffer& CurrentSwapchainFrameBuffer();

	DescriptorSetHandler& DescriptorSetHandler();

	VkDescriptorSet& CurrentDescriptorSet();

	VkCommandBuffer BeginSingleTimeCommands();

	void EndSingleTimeCommands(VkCommandBuffer& command_buffer);

	void CleanInitStructs();

	vector<VkSemaphore>& WaitSemaphores();

	vector<VkSemaphore>& SignalSemaphores();

	vector<VkPipelineStageFlags>& WaitStages();

	MeshPushConstants ViewportRotation();

	cudaError_t InteropDrawFrame();

	void DrawVerticesCall();

	VkDevice device_;
	VkPhysicalDevice physical_device_;

	VkSurfaceKHR surface_;
	VkSurfaceFormatKHR surface_format_;

	VkSwapchainKHR swap_chain_;
	bool swap_chain_rebuilding_ = false;

	VkRenderPass render_pass_;

	vector<VkCommandBuffer> command_buffers_;

	VkCommandPool command_pool_;
	VkDescriptorPoolCreateInfo pool_info_ = {};
	VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

	VkPipeline render_pipeline_;
	VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
	VkPipelineLayout mesh_pipeline_layout_ = VK_NULL_HANDLE;

	uint32_t queue_family_ = (uint32_t) - 1;
	VkQueue queue_ = VK_NULL_HANDLE;

	vector<VkSemaphore> image_semaphores_, render_semaphores_;
	vector<VkFence> render_fences_;

	VkViewport viewport_;
	VkRect2D scissor_;
	VkExtent2D extent_;

	VkPresentModeKHR present_mode_;

	vector<VkClearValue> clear_values_;

	uint32_t image_index_ = 0, current_frame_ = 0, frame_index_ = 0;

	uint32_t MAX_FRAMES_IN_FLIGHT_;

	uint2 size_;

	VertexData mesh_data_;
	ShaderLoader shader_handler_;
	SwapChainProperties swap_chain_helper_;
	MeshViewport mesh_viewport_;
	SyncStruct sync_struct_;
	CudaInterop interop_handler_;

	Grid* grid_;

private:
	void SwapchainInit();
	void SwapchainInitStage2();

	void ViewportInit();
	void RenderpassInit();

	void BulkHelperStructInit();

	void InFlightObjectsInit();

	void SimulationInit();

	VkResult PipelineInit();

	VulkanHelper vulkan_helper_;
	RenderPassInitializer render_pass_initializer_;
};