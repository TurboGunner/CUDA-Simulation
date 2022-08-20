#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <vulkan/vulkan.h>
#include "gui_driver.hpp"

#include <algorithm>
#include <limits> 
#include <vector>

using std::vector;

class SwapChainProperties {
public:
    SwapChainProperties(VkDevice device_in, VkPhysicalDevice& phys_device_in) {
        device_ = device_in;
        physical_device_ = phys_device_in;
    }

    void CreateSwapChain(VkSurfaceKHR& surface) {
        VkSurfaceFormatKHR surfaceFormat = ChooseSwapSurfaceFormat();

        uint32_t image_count = capabilities_.minImageCount + 1;
        if (capabilities_.maxImageCount > 0 && image_count > capabilities_.maxImageCount) {
            image_count = capabilities_.maxImageCount;
        }

        VkSwapchainCreateInfoKHR create_info = SwapChainInfo(surface, image_count);

        VkSwapchainKHR swap_chain;

        if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swap_chain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }
    }

    VkSwapchainCreateInfoKHR SwapChainInfo(VkSurfaceKHR& surface, uint32_t image_count) {
        VkSwapchainCreateInfoKHR create_info {};

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

    void InitializeSurfaceCapabilities(VkSurfaceKHR& surface) {
        vk_status = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface, &capabilities_);
        VulkanErrorHandler(vk_status);

        uint32_t formatCount;
        vk_status = vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface, &formatCount, nullptr);
        VulkanErrorHandler(vk_status);

        if (formatCount != 0) {
            formats_.resize(formatCount);
            vk_status = vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface, &formatCount, formats_.data());
            VulkanErrorHandler(vk_status);
        }

        uint32_t present_mode_count;
        vk_status = vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface, &present_mode_count, nullptr);
        VulkanErrorHandler(vk_status);

        if (present_mode_count != 0) {
             present_modes_.resize(present_mode_count);
            vk_status = vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface, &present_mode_count,  present_modes_.data());
            VulkanErrorHandler(vk_status);
        }
    }

    VkSurfaceFormatKHR ChooseSwapSurfaceFormat() {
        for (const auto& availableFormat : formats_) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }
        return formats_[0];
    }

    VkPresentModeKHR ChooseSwapPresentMode() {
        for (const auto& availablePresentMode :  present_modes_) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D ChooseSwapExtent(uint2 size) {
        if (capabilities_.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities_.currentExtent;
        }
        VkExtent2D actualExtent = {
            static_cast<uint32_t>(size.x),
            static_cast<uint32_t>(size.y)
        };

        actualExtent.width = std::clamp(actualExtent.width, capabilities_.minImageExtent.width, capabilities_.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities_.minImageExtent.height, capabilities_.maxImageExtent.height);

        return actualExtent;
    }

    VkDevice device_;
    VkPhysicalDevice physical_device_;

    VkSurfaceCapabilitiesKHR capabilities_;

    vector<VkSurfaceFormatKHR> formats_;
    vector<VkPresentModeKHR> present_modes_;

    VkSurfaceFormatKHR surface_format_;
    VkPresentModeKHR present_mode_;

    VkExtent2D extent_;

    vector<VkImage> swap_chain_images_;

    VkResult vk_status;
};