#include "gui_driver.hpp"


void VulkanGUIDriver::LoadInitializationInfo(ImGui_ImplVulkan_InitInfo& init_info, ImGui_ImplVulkanH_Window* window) {
    init_info.Instance = instance_;
    init_info.PhysicalDevice = physical_device_;
    init_info.Device = device_;

    init_info.QueueFamily = queue_family_;
    init_info.Queue = queue_;

    init_info.PipelineCache = pipeline_cache_;

    init_info.DescriptorPool = descriptor_pool_;

    init_info.Subpass = 0;

    init_info.MinImageCount = min_image_count_;
    init_info.ImageCount = window->ImageCount;

    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    init_info.Allocator = allocators_;
    init_info.CheckVkResultFn = VulkanErrorHandler;
}

void VulkanGUIDriver::CreateFrameBuffers(int& width, int& height, VkSurfaceKHR& surface) {
    SDL_GetWindowSize(window, &width, &height);
    wd_ = &main_window_data_;
    SetupVulkanWindow(surface, width, height);
}

void VulkanGUIDriver::FramePresent() {
    if (swap_chain_rebuilding_) {
        return;
    }
    VkSemaphore render_complete_semaphore = wd_->FrameSemaphores[wd_->SemaphoreIndex].RenderCompleteSemaphore;
    VkPresentInfoKHR info = {};

    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &render_complete_semaphore;

    info.swapchainCount = 1;

    info.pSwapchains = &wd_->Swapchain;
    info.pImageIndices = &wd_->FrameIndex;

    vulkan_status = vkQueuePresentKHR(queue_, &info);
    if (vulkan_status == VK_ERROR_OUT_OF_DATE_KHR || vulkan_status == VK_SUBOPTIMAL_KHR)
    {
        swap_chain_rebuilding_ = true;
        return;
    }
    VulkanErrorHandler(vulkan_status);
    wd_->SemaphoreIndex = (wd_->SemaphoreIndex + 1) % wd_->ImageCount;
}

void VulkanGUIDriver::SwapChainCondition() {
    if (!swap_chain_rebuilding_) {
        return;
    }
    int width, height;
    SDL_GetWindowSize(window, &width, &height);
    if (width > 0 && height > 0) {
        ImGui_ImplVulkan_SetMinImageCount(min_image_count_);
        ImGui_ImplVulkanH_CreateOrResizeWindow(instance_, physical_device_, device_, &main_window_data_, queue_family_, allocators_, width, height, min_image_count_);

        main_window_data_.FrameIndex = 0;
        swap_chain_rebuilding_ = false;
    }
}

void VulkanGUIDriver::ManageCommandBuffer(ImGui_ImplVulkanH_Frame* frame_draw) {
    vulkan_status = vkResetCommandPool(device_, frame_draw->CommandPool, 0);
    VulkanErrorHandler(vulkan_status);

    VkCommandBufferBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vulkan_status = vkBeginCommandBuffer(frame_draw->CommandBuffer, &info);
    VulkanErrorHandler(vulkan_status);
}