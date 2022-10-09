#include "gui_driver.cuh"

void VulkanGUIDriver::DebugErrorCallback() {
    debug_info_callback_.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
    debug_info_callback_.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;

    debug_info_callback_.pfnCallback = DebugReport;
    debug_info_callback_.pUserData = NULL;

    vulkan_status = InstanceDebugCallbackEXT(instance_, &debug_info_callback_, allocators_, &debug_report_callback_);
    VulkanErrorHandler(vulkan_status);

    ProgramLog::OutputLine("\nShout out to Bidoof on the beat!\n");
}

void VulkanGUIDriver::DebugOptionInitialization(const char** extensions, const uint32_t& ext_count) {
    // Enabling validation layers
    const char* layers[] = { "VK_LAYER_KHRONOS_validation" };
    instance_info_.enabledLayerCount = 1;
    instance_info_.ppEnabledLayerNames = layers;

    // Enable debug report extension (we need additional storage, so we duplicate the user array to add our new extension to it)
    const char** extensions_ext = (const char**) malloc(sizeof(const char*) * (ext_count + 1));
    memcpy(extensions_ext, extensions, ext_count * sizeof(const char*));

    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "CUDA Sim";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 3, 2);
    app_info.pEngineName = "BloodFlow";
    app_info.engineVersion = VK_MAKE_VERSION(1, 3, 2);
    app_info.apiVersion = VK_MAKE_VERSION(1, 3, 2);

    extensions_ext[ext_count] = "VK_EXT_debug_report";
    instance_info_.enabledExtensionCount = ext_count + 1;
    instance_info_.ppEnabledExtensionNames = extensions_ext;
    instance_info_.pApplicationInfo = &app_info;

    // Create Vulkan Instance
    vulkan_status = vkCreateInstance(&instance_info_, allocators_, &instance_);

    VulkanErrorHandler(vulkan_status);
    free(extensions_ext);

    // Get the function pointer (required for any extensions)
    InstanceDebugCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)
        vkGetInstanceProcAddr(instance_, "vkCreateDebugReportCallbackEXT");

    IM_ASSERT(InstanceDebugCallbackEXT != NULL);

    s_stream << "Address for Vulkan debug callback: " << InstanceDebugCallbackEXT << ". Debug callback successful!";
    ProgramLog::OutputLine(s_stream);

    DebugErrorCallback();
}