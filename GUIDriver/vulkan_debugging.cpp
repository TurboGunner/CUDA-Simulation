#include "gui_driver.cuh"

VkResult VulkanGUIDriver::DebugErrorCallback() {
#ifdef IMGUI_VULKAN_DEBUG_REPORT
    debug_info_callback_.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
    debug_info_callback_.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;

    debug_info_callback_.pfnCallback = DebugReport;
    debug_info_callback_.pUserData = nullptr;
#endif

    VkResult vulkan_status = InstanceDebugCallbackEXT(instance_, &debug_info_callback_, allocators_, &debug_report_callback_);
    VulkanErrorHandler(vulkan_status);

    ProgramLog::OutputLine("\nShout out to Bidoof on the beat!\n");
    return vulkan_status;
}

VkResult VulkanGUIDriver::DebugOptionInitialization() {
    // Enabling validation layers

    // Enable debug report extension (we need additional storage, so we duplicate the user array to add our new extension to it)

    VkApplicationInfo app_info = {};

    vector<const char*> layers = { "VK_LAYER_KHRONOS_validation" };
    instance_info_.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info_.enabledLayerCount = 1;
    instance_info_.ppEnabledLayerNames = layers.data();

    uint32_t count;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr); //get number of extensions
    vector<VkExtensionProperties> extensions(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data()); //populate buffer

    ProgramLog::OutputLine("Vulkan Physical Device Extension Count: " + std::to_string(count));

    for (auto& extension_properties : extensions) {
        s_stream << "Extension Name: " << extension_properties.extensionName << std::endl;
    }
    ProgramLog::OutputLine(s_stream);

    // Create Vulkan Instance
    VkResult vulkan_status = vkCreateInstance(&instance_info_, allocators_, &instance_);

    VulkanErrorHandler(vulkan_status);

    // Get the function pointer (required for any extensions)
    InstanceDebugCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT) vkGetInstanceProcAddr(instance_, "vkCreateDebugReportCallbackEXT");

    if (!InstanceDebugCallbackEXT) {
        ProgramLog::OutputLine("Error: Debug callback was not able to be generated!");
        throw std::system_error(ENXIO, std::generic_category(), "Debug callback failed to generate.");
    }

    s_stream << "Address for Vulkan debug callback: " << InstanceDebugCallbackEXT << ". Debug callback successful!";
    ProgramLog::OutputLine(s_stream);

    vulkan_status = DebugErrorCallback();

    return vulkan_status;
}