#include "swap_chain_manager.cuh"

void SwapChainProperties::AllocateImages(uint32_t image_count) {
    vulkan_status = vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr);
    swapchain_images_.resize(image_count);
    vulkan_status = vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, swapchain_images_.data());
}

VkSwapchainCreateInfoKHR SwapChainProperties::SwapChainInfo(VkSurfaceKHR& surface, uint32_t image_count) {
    VkSwapchainCreateInfoKHR create_info = {};

    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = surface;

    create_info.minImageCount = image_count;
    create_info.imageFormat = surface_format_.format;
    create_info.imageColorSpace = surface_format_.colorSpace;
    create_info.imageExtent = extent_;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;

    create_info.preTransform = capabilities_.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = present_mode_;
    create_info.clipped = VK_TRUE;

    create_info.oldSwapchain = VK_NULL_HANDLE;

    return create_info;
}

void SwapChainProperties::InitializeSurfaceCapabilities(VkSurfaceKHR& surface) {
    vulkan_status = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface, &capabilities_);

    uint32_t format_count;
    vulkan_status = vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface, &format_count, nullptr);

    if (format_count != 0) {
        formats_.resize(format_count);
        vulkan_status = vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface, &format_count, formats_.data());
    }

    uint32_t present_mode_count;
    vulkan_status = vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface, &present_mode_count, nullptr);

    if (present_mode_count != 0) {
        present_modes_.resize(present_mode_count);
        vulkan_status = vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface, &present_mode_count, present_modes_.data());
    }
}

VkSurfaceFormatKHR SwapChainProperties::ChooseSwapSurfaceFormat() {
    for (const auto& available_format : formats_) {
        if (available_format.format == VK_FORMAT_B8G8R8A8_SRGB && available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return available_format;
        }
    }
    return formats_[0];
}

VkPresentModeKHR SwapChainProperties::ChooseSwapPresentMode() {
    for (const auto& available_present_mode : present_modes_) {
        if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return available_present_mode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D SwapChainProperties::ChooseSwapExtent() {
    if (capabilities_.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities_.currentExtent;
    }
    VkExtent2D actual_extent = { size_.x, size_.y };

    actual_extent.width = ClampNum(actual_extent.width, capabilities_.minImageExtent.width, capabilities_.maxImageExtent.width);
    actual_extent.height = ClampNum(actual_extent.height, capabilities_.minImageExtent.height, capabilities_.maxImageExtent.height);

    return actual_extent;
}

void SwapChainProperties::CreateImageViews() {
    swapchain_image_views_.resize(swapchain_images_.size());

    for (uint32_t i = 0; i < swapchain_images_.size(); i++) {
        swapchain_image_views_[i] = ImageHelper::CreateImageView(device_, swapchain_images_[i], surface_format_.format, VK_IMAGE_ASPECT_COLOR_BIT);
    }
}

VkImageView SwapChainProperties::DepthImageView() {

    auto format = ImageHelper::FindDepthFormat(physical_device_);

    ImageHelper::InitializeImage(device_, physical_device_, depth_memory_, depth_image_, size_, format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_TYPE_2D);

    auto depth_image_view = ImageHelper::CreateImageView(device_, depth_image_, format, VK_IMAGE_ASPECT_DEPTH_BIT);

    ProgramLog::OutputLine("Created depth image view.");

    return depth_image_view;
}