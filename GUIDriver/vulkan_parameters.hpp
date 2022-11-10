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

	VkResult RebuildSwapchain();

	void ManipulateCamera() {
		mesh_viewport_.ManipulateCamera();
	}

	const vector<const char*>& InteropDeviceExtensions() const {
		return interop_handler_.interop_device_extensions_;
	}

	const vector<const char*>& InteropExtensions() const {
		return interop_handler_.interop_extensions_;
	}

	const size_t SwapchainImagesSize() const {
		return swap_chain_helper_.swapchain_images_.size();
	}

	void BindMeshPipeline() {
		mesh_data_.BindPipeline(InFlightCommandBuffer(), command_pool_);
	}

	VkDevice device_;
	VkPhysicalDevice physical_device_;

	VkSurfaceKHR surface_;
	VkSurfaceFormatKHR surface_format_;

	VkSwapchainKHR swap_chain_;

	VkRenderPass render_pass_;

	vector<VkCommandBuffer> command_buffers_;

	VkCommandPool command_pool_;
	VkDescriptorPoolCreateInfo pool_info_ = {};
	VkDescriptorPool descriptor_pool_ = nullptr;

	VkPipeline render_pipeline_ = nullptr;
	VkPipelineCache pipeline_cache_ = nullptr;
	VkPipelineLayout mesh_pipeline_layout_ = nullptr;

	uint32_t queue_family_ = (uint32_t) - 1;
	VkQueue queue_ = nullptr;

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

	Grid* grid_;

	bool swap_chain_rebuilding_ = false;

	SyncStruct sync_struct_;

private:
	VertexData mesh_data_;
	ShaderLoader shader_handler_;
	SwapChainProperties swap_chain_helper_;
	MeshViewport mesh_viewport_;
	CudaInterop interop_handler_;

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