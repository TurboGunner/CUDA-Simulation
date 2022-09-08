#include "gui_driver.cuh"

void VulkanGUIDriver::SetupVulkanWindow(VkSurfaceKHR& surface, int width, int height) {
    VkBool32 res;
    vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_, queue_family_, surface_, &res);
    if (res != VK_TRUE) {
        ProgramLog::OutputLine("Error: no WSI support on physical device 0!");
    }

    IM_ASSERT(min_image_count_ >= 2);
}

void VulkanGUIDriver::CleanupVulkan() {
    vkDestroyDescriptorPool(device_, descriptor_pool_, allocators_);

    swap_chain_helper_.Clean();
    sync_struct_.Clean();
    shader_handler_.Clean();
    render_pass_initializer_.Clean();
    mesh_data_.Clean();

    if (surface_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
    }

#ifdef IMGUI_VULKAN_DEBUG_REPORT
    auto vkDestroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance_, "vkDestroyDebugReportCallbackEXT");
    vkDestroyDebugReportCallbackEXT(instance_, debug_report_callback_, allocators_);
#endif
    vkDestroyDevice(device_, allocators_);
    vkDestroyInstance(instance_, allocators_);
}