#include "gui_driver.cuh"

void VulkanGUIDriver::VulkanInstantiation() {
    VkApplicationInfo app_info = {}; //NOTE: REDUNDANT

    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "CUDA + Vulkan Simulator";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 3, 2);
    app_info.pEngineName = "BloodFlow";
    app_info.engineVersion = VK_MAKE_VERSION(1, 3, 2);
    app_info.apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 205);

    instance_info_.enabledExtensionCount = phys_device_extensions_.size();
    instance_info_.ppEnabledExtensionNames = phys_device_extensions_.data();
    instance_info_.pApplicationInfo = &app_info;
#ifdef IMGUI_VULKAN_DEBUG_REPORT
    DebugOptionInitialization();
#else
    VkResult vulkan_status = vkCreateInstance(&instance_info_, allocators_, &instance_);
    VulkanErrorHandler(vulkan_status);
    IM_UNUSED(debug_report_callback_);
#endif
    SelectGPU();

    SelectQueueFamily();

    LogicalDeviceInitialization();

    PoolDescriptionInitialization();
    ProgramLog::OutputLine("Vulkan instance successfully created!");
}

void VulkanGUIDriver::SelectQueueFamily() {
    uint32_t count;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &count, nullptr);

    auto* queues = new VkQueueFamilyProperties[count];
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &count, queues);

    for (uint32_t i = 0; i < count; i++) {
        if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            queue_family_ = i;
            break;
        }
    }

    delete[] queues;

    if (queue_family_ == (uint32_t) - 1) {
        ProgramLog::OutputLine("Error: No valid queue family was found!");
        throw std::system_error(ENXIO, std::generic_category(), "No valid queue family.");
    }

    ProgramLog::OutputLine("No errors detected while selecting Vulkan queue family.");
}

VkResult VulkanGUIDriver::SelectGPU() {
    uint32_t gpu_count;

    VkResult vulkan_status = vkEnumeratePhysicalDevices(instance_, &gpu_count, nullptr);
    VulkanErrorHandler(vulkan_status);

    if (gpu_count == 0) {
        ProgramLog::OutputLine("Error: No GPU was selected!");
        throw std::system_error(ENXIO, std::generic_category(), "No GPUs were detected");
    }

    vector<VkPhysicalDevice> gpus(gpu_count);

    vulkan_status = vkEnumeratePhysicalDevices(instance_, &gpu_count, gpus.data());
    VulkanErrorHandler(vulkan_status);

    size_t use_gpu = 0;
    for (size_t i = 0; i < gpu_count; i++) {
        vkGetPhysicalDeviceProperties(gpus[i], &device_properties_);
        if (device_properties_.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            use_gpu = i;
            break;
        }
    }

    s_stream << "GPU Device Name: " << device_properties_.deviceName;
    ProgramLog::OutputLine(s_stream);

    physical_device_ = gpus[use_gpu];
    s_stream << "GPU Device #" << use_gpu << " has been selected for Vulkan Rendering successfully!";
    ProgramLog::OutputLine(s_stream);

    return vulkan_status;
}

VkResult VulkanGUIDriver::PoolDescriptionInitialization() {
    const vector<VkDescriptorPoolSize> pool_sizes = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 },
    };

    pool_info_.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info_.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    //Size Properties

    size_t descriptor_size = pool_sizes.size();

    pool_info_.maxSets = 1000 * descriptor_size;
    pool_info_.poolSizeCount = descriptor_size;
    pool_info_.pPoolSizes = pool_sizes.data();

    VkResult vulkan_status = vkCreateDescriptorPool(device_, &pool_info_, allocators_, &descriptor_pool_);
    VulkanErrorHandler(vulkan_status);

    return vulkan_status;
}

VkResult VulkanGUIDriver::LogicalDeviceInitialization() {
    vector<const char*> interop_device_extensions = vulkan_parameters_.InteropDeviceExtensions();

    device_extensions_.insert(device_extensions_.begin(), interop_device_extensions.begin(), interop_device_extensions.end());

    ProgramLog::OutputLine("Vulkan Device Extension Count: " + std::to_string(device_extensions_.size()));

    const float queue_priority = 1.0f;

    VkDeviceQueueCreateInfo queue_info = {};

    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_info = {};

    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = device_extensions_.size();
    device_info.ppEnabledExtensionNames = device_extensions_.data();

    VkResult vulkan_status = vkCreateDevice(physical_device_, &device_info, allocators_, &device_);

    VulkanErrorHandler(vulkan_status);
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

    return vulkan_status;
}