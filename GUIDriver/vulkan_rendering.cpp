#include "gui_driver.cuh"

VkResult VulkanGUIDriver::SetupVulkanWindow() {
    VkBool32 result;
    VkResult vulkan_status = vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_, queue_family_, surface_, &result);
    if (!result) {
        ProgramLog::OutputLine("Error: no WSI support on physical device 0!");
    }

    if (min_image_count_ < 2) {
        ProgramLog::OutputLine("Error: The minimum image count is too low. It must be a minimum of 2.");
        throw std::system_error(ENXIO, std::generic_category(), "Minimum image count has not been met.");
    }
    return vulkan_status;
}

void VulkanGUIDriver::CleanupVulkan() {
    vkDestroyDescriptorPool(device_, descriptor_pool_, allocators_);

    vulkan_parameters_.CleanInitStructs();

    if (surface_) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
    }

#ifdef IMGUI_VULKAN_DEBUG_REPORT
    auto vkDestroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT) vkGetInstanceProcAddr(instance_, "vkDestroyDebugReportCallbackEXT");
    vkDestroyDebugReportCallbackEXT(instance_, debug_report_callback_, allocators_);
#endif
    vkDestroyDevice(device_, allocators_);
    vkDestroyInstance(instance_, allocators_);
}