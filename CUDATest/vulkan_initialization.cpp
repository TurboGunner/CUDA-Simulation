#include "gui_driver.hpp"

#include "vulkan/vulkan.hpp"

void VulkanGUIDriver::LoadInstanceProperties(const char** extensions, const uint32_t& ext_count) {
    instance_info_.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info_.enabledExtensionCount = ext_count;
    instance_info_.ppEnabledExtensionNames = extensions;
}

void VulkanGUIDriver::VulkanInitialization(const char** extensions, uint32_t ext_count) {
    // Create Vulkan Instance

    vk::DynamicLoader dl;

    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    vk::Instance instance = vk::createInstance({}, nullptr);

    // initialize function pointers for instance
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);

    // create a dispatcher, based on additional vkDevice/vkGetDeviceProcAddr
    std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
    assert(!physicalDevices.empty());

    vk::Device device = physicalDevices[0].createDevice({}, nullptr);

    // function pointer specialization for device
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);

    LoadInstanceProperties(extensions, ext_count);

#ifdef IMGUI_VULKAN_DEBUG_REPORT
    DebugOptionInitialization(extensions, ext_count);
#else
    vulkan_status = vkCreateInstance(&instance_info_, allocators_, &instance_);
    VulkanErrorHandler(vulkan_status);
    IM_UNUSED(debug_report_callback_);
#endif
    SelectGPU();

    SelectQueueFamily();

    LogicalDeviceInitialization();

    PoolDescriptionInitialization();
}

void VulkanGUIDriver::SelectQueueFamily() {
    uint32_t count;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &count, NULL);

    auto* queues = (VkQueueFamilyProperties*)malloc(sizeof(VkQueueFamilyProperties) * count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &count, queues);

    for (uint32_t i = 0; i < count; i++)
        if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            queue_family_ = i;
            break;
        }

    free(queues);
    IM_ASSERT(queue_family_ != (uint32_t)-1);
}

void VulkanGUIDriver::SelectGPU() {
    uint32_t gpu_count;

    vulkan_status = vkEnumeratePhysicalDevices(instance_, &gpu_count, NULL);
    VulkanErrorHandler(vulkan_status);

    IM_ASSERT(gpu_count > 0);

    VkPhysicalDevice* gpus = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * gpu_count);
    vulkan_status = vkEnumeratePhysicalDevices(instance_, &gpu_count, gpus);

    VulkanErrorHandler(vulkan_status);

    int use_gpu = 0;
    for (size_t i = 0; i < gpu_count; i++) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(gpus[i], &properties);
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            use_gpu = i;
            break;
        }
    }
    physical_device_ = gpus[use_gpu];
    free(gpus);
}

void VulkanGUIDriver::PoolDescriptionInitialization() {
    VkDescriptorPoolSize pool_sizes[] =
    {
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
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };

    LoadPoolDescriptionProperties(pool_info_, pool_sizes);

    vulkan_status = vkCreateDescriptorPool(device_, &pool_info_, allocators_, &descriptor_pool_);
    VulkanErrorHandler(vulkan_status);
}

void VulkanGUIDriver::LoadPoolDescriptionProperties(VkDescriptorPoolCreateInfo& pool_info_, VkDescriptorPoolSize pool_sizes[]) {
    //Type Properties
    pool_info_.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info_.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    //Size Properties
    pool_info_.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
    pool_info_.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
    pool_info_.pPoolSizes = pool_sizes;
}

void VulkanGUIDriver::LogicalDeviceInitialization() {
    int device_extension_count = 1;
    const char* device_extensions[] = { "VK_KHR_swapchain" };
    const float queue_priority[] = { 1.0f };

    VkDeviceQueueCreateInfo queue_info[1] = {};
    queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info[0].queueFamilyIndex = queue_family_;
    queue_info[0].queueCount = 1;
    queue_info[0].pQueuePriorities = queue_priority;

    VkDeviceCreateInfo instance_info_ = {};

    instance_info_.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    instance_info_.queueCreateInfoCount = sizeof(queue_info) / sizeof(queue_info[0]);
    instance_info_.pQueueCreateInfos = queue_info;
    instance_info_.enabledExtensionCount = device_extension_count;
    instance_info_.ppEnabledExtensionNames = device_extensions;

    vulkan_status = vkCreateDevice(physical_device_, &instance_info_, allocators_, &device_);

    VulkanErrorHandler(vulkan_status);
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
}