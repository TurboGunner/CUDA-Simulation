#include "gui_driver.cuh"

void VulkanGUIDriver::LoadInitializationInfo(ImGui_ImplVulkan_InitInfo& init_info) {
    init_info.Instance = instance_;
    init_info.PhysicalDevice = physical_device_;
    init_info.Device = device_;

    init_info.QueueFamily = queue_family_;
    init_info.Queue = queue_;

    init_info.PipelineCache = vulkan_parameters_.pipeline_cache_;

    init_info.DescriptorPool = descriptor_pool_;

    init_info.MinImageCount = MAX_FRAMES_IN_FLIGHT_;
    init_info.ImageCount = vulkan_parameters_.SwapchainImagesSize();

    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    init_info.Allocator = allocators_;
    init_info.CheckVkResultFn = VulkanErrorHandler;
}

void VulkanGUIDriver::PresentFrame() {
    if (vulkan_parameters_.swap_chain_rebuilding_) {
        return;
    }

    VkPresentInfoKHR info = {};

    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &vulkan_parameters_.InFlightRenderSemaphore();

    info.swapchainCount = 1;

    info.pSwapchains = &vulkan_parameters_.swap_chain_;
    info.pImageIndices = &vulkan_parameters_.image_index_;

    VkResult vulkan_status = vkQueuePresentKHR(vulkan_parameters_.queue_, &info);
    if (vulkan_status == VK_ERROR_OUT_OF_DATE_KHR || vulkan_status == VK_SUBOPTIMAL_KHR) {
        vulkan_parameters_.swap_chain_rebuilding_ = true;
        return;
    }
    VulkanErrorHandler(vulkan_status);
    vulkan_parameters_.current_frame_ = (vulkan_parameters_.current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT_;
}

void VulkanGUIDriver::SwapChainCondition() {
    if (!vulkan_parameters_.swap_chain_rebuilding_) {
        return;
    }
    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    if (width > 0 && height > 0) {
        ImGui_ImplVulkan_SetMinImageCount(min_image_count_);
        vulkan_parameters_.swap_chain_rebuilding_ = false;
    }

    vulkan_parameters_.RebuildSwapchain();
}

void VulkanGUIDriver::InitializeVulkan() {
    uint32_t ext_count = 0;
    SDL_Vulkan_GetInstanceExtensions(window, &ext_count, nullptr);

    auto& interop_extensions = vulkan_parameters_.InteropExtensions();

    ext_count += interop_extensions.size();

    phys_device_extensions_.insert(phys_device_extensions_.begin(), interop_extensions.begin(), interop_extensions.end());

    const char** sdl_extensions = new const char*[ext_count];

    SDL_Vulkan_GetInstanceExtensions(window, &ext_count, sdl_extensions);

    for (int i = 0; i < ext_count; i++) {
        phys_device_extensions_.push_back(sdl_extensions[i]);
    }

    if (ext_count > 0) {
        delete[] sdl_extensions;
    }

    VulkanInstantiation();
}