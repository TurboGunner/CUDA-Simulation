#pragma once

#include "gui_driver.cuh"
#include "vulkan_parameters.hpp"

#include "../CUDATest/handler_classes.hpp"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>

using std::tuple;
using std::vector;

class SwapChainProperties {
public:
    SwapChainProperties() = default;

    SwapChainProperties(VkDevice& device_in, VkPhysicalDevice& phys_device_in) {
        device_ = device_in;
        physical_device_ = phys_device_in;
    }

    static uint32_t ClampNum(const uint32_t& value, const uint32_t& min, const uint32_t& max) {
        return std::max(min, std::min(max, value));
    }

    VkSwapchainKHR Initialize(VkSurfaceKHR& surface, uint2 size) {
        InitializeSurfaceCapabilities(surface);
        surface_format_ = ChooseSwapSurfaceFormat();

        uint32_t image_count = capabilities_.minImageCount + 1;
        if (capabilities_.maxImageCount > 0 && image_count > capabilities_.maxImageCount) {
            image_count = capabilities_.maxImageCount;
        }

        extent_ = ChooseSwapExtent(size);

        present_mode_ = ChooseSwapPresentMode();

        VkSwapchainCreateInfoKHR create_info = SwapChainInfo(surface, image_count);

        if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swapchain_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        AllocateImages(image_count);

        return swapchain_;
    }

    VkSwapchainKHR swapchain_;

    VkExtent2D extent_;

    vector<VkImage> swapchain_images_;
    vector<VkImageView> swapchain_image_views_;

    VkSurfaceFormatKHR surface_format_;
    VkPresentModeKHR present_mode_;

private:
    void AllocateImages(uint32_t image_count) {
        vulkan_status = vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr);
        swapchain_images_.resize(image_count);
        vulkan_status = vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, swapchain_images_.data());
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
        vulkan_status = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface, &capabilities_);

        uint32_t format_count;
        vulkan_status = vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface, &format_count, nullptr);

        if (format_count != 0) {
            formats_.resize(format_count);
            vulkan_status = vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface, &format_count, formats_.data());
        }

        s_stream << "\nSwapchain Format Count: " << formats_.size() << "\n";
        ProgramLog::OutputLine(s_stream);

        uint32_t present_mode_count;
        vulkan_status = vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface, &present_mode_count, nullptr);

        if (present_mode_count != 0) {
             present_modes_.resize(present_mode_count);
            vulkan_status = vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface, &present_mode_count,  present_modes_.data());
        }
    }

    VkSurfaceFormatKHR ChooseSwapSurfaceFormat() {
        for (const auto& available_format : formats_) {
            if (available_format.format == VK_FORMAT_B8G8R8A8_SRGB && available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return available_format;
            }
        }
        return formats_[0];
    }

    VkPresentModeKHR ChooseSwapPresentMode() {
        for (const auto& available_present_mode :  present_modes_) {
            if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return available_present_mode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D ChooseSwapExtent(uint2 size) {
        if (capabilities_.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities_.currentExtent;
        }
        VkExtent2D actual_extent = {
            static_cast<uint32_t>(size.x),
            static_cast<uint32_t>(size.y)
        };

        actual_extent.width = ClampNum(actual_extent.width, capabilities_.minImageExtent.width, capabilities_.maxImageExtent.width);
        actual_extent.height = ClampNum(actual_extent.height, capabilities_.minImageExtent.height, capabilities_.maxImageExtent.height);

        return actual_extent;
    }

    VkDevice device_;
    VkPhysicalDevice physical_device_;

    VkSurfaceCapabilitiesKHR capabilities_;

    vector<VkSurfaceFormatKHR> formats_;
    vector<VkPresentModeKHR> present_modes_;

    VkResult vulkan_status;
};