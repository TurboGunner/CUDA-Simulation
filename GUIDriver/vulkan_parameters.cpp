#include "vulkan_parameters.hpp"

VulkanParameters::VulkanParameters(VkDevice& device_in, VkPhysicalDevice& phys_device_in, VkSurfaceKHR& surface_in, const uint32_t& max_frames_in_flight_in, VkQueue& queue_in, const uint32_t& queue_family_in, VkDescriptorPool& descriptor_pool_in, uint2& size_in) {
	size_ = size_in;
	device_ = device_in;
	physical_device_ = phys_device_in;
	surface_ = surface_in;

	MAX_FRAMES_IN_FLIGHT_ = max_frames_in_flight_in;

	queue_ = queue_in;
	queue_family_ = queue_family_in;

	descriptor_pool_ = descriptor_pool_in;
}

VkResult VulkanParameters::InitVulkan() {
	SwapchainInit();
	ViewportInit();
	RenderpassInit();

	BulkHelperStructInit();

	vulkan_helper_.CreateCommandPool(command_pool_, queue_family_);

	shader_handler_.InitializeDescriptorHandler(physical_device_, descriptor_pool_, command_pool_, queue_, MAX_FRAMES_IN_FLIGHT_);

	SwapchainInitStage2();

	InFlightObjectsInit();

	VkResult vulkan_status = PipelineInit();
	return vulkan_status;
}

VkResult VulkanParameters::InitVulkanStage2() {
	VkResult vulkan_status = vkDeviceWaitIdle(device_);
	return vulkan_status;
}

void VulkanParameters::SwapchainInit() {
	swap_chain_helper_ = SwapChainProperties(device_, physical_device_, surface_, queue_, size_);
	swap_chain_ = swap_chain_helper_.Initialize();

	surface_format_ = swap_chain_helper_.surface_format_;
	present_mode_ = swap_chain_helper_.present_mode_;
	extent_ = swap_chain_helper_.extent_;
}

void VulkanParameters::SwapchainInitStage2() {
	//Creating depth pass for rendering
	swap_chain_helper_.InitializeDepthPass(command_pool_);

	//Creating framebuffers
	ProgramLog::OutputLine("\nSwapchain Image View Size: " + std::to_string(swap_chain_helper_.swapchain_image_views_.size()));

	swap_chain_helper_.CreateSwapchainFrameBuffers(render_pass_, swap_chain_helper_.swapchain_image_views_, swap_chain_helper_.depth_image_view_);
}

void VulkanParameters::ViewportInit() {
	viewport_.x = 0.0f;
	viewport_.y = 0.0f;
	viewport_.width = (float)size_.x;
	viewport_.height = (float)size_.y;
	viewport_.minDepth = 0.0f;
	viewport_.maxDepth = 1.0f;

	scissor_.offset = { 0, 0 };
	scissor_.extent = extent_;
}

void VulkanParameters::RenderpassInit() {
	render_pass_initializer_ = RenderPassInitializer(device_);
	render_pass_ = render_pass_initializer_.Initialize(surface_format_.format, ImageHelper::FindDepthFormat(physical_device_));
}

void VulkanParameters::BulkHelperStructInit() {
	vulkan_helper_ = VulkanHelper(device_, size_, MAX_FRAMES_IN_FLIGHT_);
	shader_handler_ = ShaderLoader(device_, viewport_, scissor_, pipeline_cache_);
	sync_struct_ = SyncStruct(device_, MAX_FRAMES_IN_FLIGHT_);
	mesh_data_ = VertexData(device_, physical_device_, queue_, MAX_FRAMES_IN_FLIGHT_);
	mesh_viewport_ = MeshViewport(device_);
	interop_handler_ = CudaInterop(device_, physical_device_);
}

void VulkanParameters::InFlightObjectsInit() {
	if (vulkan_helper_.InitializeCommandBuffers(command_buffers_, command_pool_) != VK_SUCCESS) {
		ProgramLog::OutputLine("Error: Failed to create command buffers!");
		return;
	}

	//Creating synchronization structs (fences and semaphores)
	if (sync_struct_.Initialize() != VK_SUCCESS) {
		ProgramLog::OutputLine("Error: Failed to create synchronization structs!");
		return;
	}
	//NOTE: TEMP until refactor overhaul
	image_semaphores_ = sync_struct_.present_semaphores_;
	render_semaphores_ = sync_struct_.render_semaphores_;

	render_fences_ = sync_struct_.fences_;

	clear_values_.resize(2);

	clear_values_[0].color = { {0.0f, 0.0f, 0.0f, 1.0f } };
	clear_values_[1].depthStencil = { 1.0f, 0 };
}

VkResult VulkanParameters::PipelineInit() {
	//Create Pipeline
	render_pipeline_ = shader_handler_.Initialize(render_pass_);

	VkResult vulkan_status = mesh_data_.Initialize(command_pool_);
	vulkan_status = mesh_data_.InitializeIndex(command_pool_);

	mesh_pipeline_layout_ = shader_handler_.mesh_pipeline_layout_;

	return vulkan_status;
}

void VulkanParameters::CleanInitStructs() {
	swap_chain_helper_.Clean();
	sync_struct_.Clean();
	shader_handler_.Clean();
	render_pass_initializer_.Clean();
	mesh_data_.Clean();
}