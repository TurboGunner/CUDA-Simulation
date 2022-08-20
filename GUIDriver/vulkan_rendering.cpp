#include "gui_driver.cuh"

void VulkanGUIDriver::SetupVulkanWindow(VkSurfaceKHR& surface, int width, int height) {
    surface_ = surface;
    VkBool32 res;
    vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_, queue_family_, surface_, &res);
    if (res != VK_TRUE) {
        ProgramLog::OutputLine("Error: no WSI support on physical device 0!");
    }

    // Select Surface Format
    const VkFormat requestSurfaceImageFormat[] = { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM };
    const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;

    surface_format_ = ImGui_ImplVulkanH_SelectSurfaceFormat(physical_device_, surface_, requestSurfaceImageFormat, (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat), requestSurfaceColorSpace);

    // Select Present Mode
#ifdef IMGUI_UNLIMITED_FRAME_RATE
    VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR };
#else
    VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_FIFO_KHR };
#endif
    present_mode_ = ImGui_ImplVulkanH_SelectPresentMode(physical_device_, surface_, &present_modes[0], IM_ARRAYSIZE(present_modes));

    IM_ASSERT(min_image_count_ >= 2);
    //Note!
    //ImGui_ImplVulkanH_CreateOrResizeWindow(instance_, physical_device_, device_, wd_, queue_family_, allocators_, width, height, min_image_count_);
}

void VulkanGUIDriver::CleanupVulkan() {
    vkDestroyDescriptorPool(device_, descriptor_pool_, allocators_);

#ifdef IMGUI_VULKAN_DEBUG_REPORT
    auto vkDestroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance_, "vkDestroyDebugReportCallbackEXT");
    vkDestroyDebugReportCallbackEXT(instance_, debug_report_callback_, allocators_);
#endif
    vkDestroyDevice(device_, allocators_);
    vkDestroyInstance(instance_, allocators_);
}