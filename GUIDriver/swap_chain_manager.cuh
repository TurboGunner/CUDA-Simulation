#pragma once

#include "gui_driver.cuh"
#include "vulkan_parameters.hpp"
#include "vulkan_helpers.hpp"
#include "image_helpers.hpp"

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

    SwapChainProperties(VkDevice& device_in, VkPhysicalDevice& phys_device_in, VkSurfaceKHR& surface_in, VkQueue& queue_in, uint2& size_in) {
        device_ = device_in;
        physical_device_ = phys_device_in;
        surface_ = surface_in;
        queue_ = queue_in;

        size_ = size_in;
    }

    static uint32_t ClampNum(const uint32_t& value, const uint32_t& min, const uint32_t& max) {
        return std::max(min, std::min(max, value));
    }

    VkSwapchainKHR Initialize() {
        InitializeSurfaceCapabilities(surface_);
        surface_format_ = ChooseSwapSurfaceFormat();

        uint32_t image_count = capabilities_.minImageCount + 1;
        if (capabilities_.maxImageCount > 0 && image_count > capabilities_.maxImageCount) {
            image_count = capabilities_.maxImageCount;
        }

        extent_ = ChooseSwapExtent();

        present_mode_ = ChooseSwapPresentMode();

        VkSwapchainCreateInfoKHR create_info = SwapChainInfo(surface_, image_count);

        if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swapchain_) != VK_SUCCESS) {
            ProgramLog::OutputLine("\n\nError: Failed to create swapchain!\n\n");
        }

        AllocateImages(image_count);

        s_stream << "\nSwapchain Image Count: " << swapchain_images_.size() << "\n";
        ProgramLog::OutputLine(s_stream);

        CreateImageViews();

        return swapchain_;
    }

    void InitializeDepthPass(VkCommandPool& command_pool) {
        depth_image_view_ = DepthImageView();

        ImageHelper::TransitionImageLayout(device_, command_pool, queue_, depth_image_, surface_format_.format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }

    void CreateSwapchainFrameBuffers(VkRenderPass& render_pass, vector<VkImageView>& swapchain_image_views, VkImageView& depth_image_view) {
        frame_buffers_.resize(swapchain_image_views.size());

        VkFramebufferCreateInfo frame_buffer_info = {};
        frame_buffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        frame_buffer_info.pNext = nullptr;

        frame_buffer_info.renderPass = render_pass;
        frame_buffer_info.layers = 1;

        frame_buffer_info.width = size_.x;
        frame_buffer_info.height = size_.y;

        ProgramLog::OutputLine("Framebuffers size: " + std::to_string(size_.x) + " X " + std::to_string(size_.y) + ".");

        for (size_t i = 0; i < swapchain_image_views.size(); i++) {
            array<VkImageView, 2> attachments = { swapchain_image_views[i], depth_image_view };

            frame_buffer_info.attachmentCount = attachments.size();
            frame_buffer_info.pAttachments = attachments.data();

            vulkan_status = vkCreateFramebuffer(device_, &frame_buffer_info, nullptr, &frame_buffers_[i]);
        }
    }

    void RecreateSwapChain(VkRenderPass& render_pass, VkCommandPool& command_pool) {
        vkDeviceWaitIdle(device_);

        Clean();

        Initialize();
        CreateImageViews();
        CreateSwapchainFrameBuffers(render_pass, swapchain_image_views_, depth_image_view_);
        InitializeDepthPass(command_pool);
    }

    void Clean() {
        vkDestroyImageView(device_, depth_image_view_, nullptr);
        vkDestroyImage(device_, depth_image_, nullptr);
        vkFreeMemory(device_, depth_memory_, nullptr);

        for (auto frame_buffer : frame_buffers_) {
            vkDestroyFramebuffer(device_, frame_buffer, nullptr);
        }

        for (auto view : swapchain_image_views_) {
            vkDestroyImageView(device_, view, nullptr);
        }

        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    }

    VkSwapchainKHR swapchain_;

    VkExtent2D extent_;

    VkImage depth_image_;
    VkImageView depth_image_view_;

    vector<VkImage> swapchain_images_;
    vector<VkImageView> swapchain_image_views_;

    vector<VkFramebuffer> frame_buffers_;

    VkSurfaceFormatKHR surface_format_;
    VkPresentModeKHR present_mode_;

private:
    void AllocateImages(uint32_t image_count) {
        vulkan_status = vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr);
        swapchain_images_.resize(image_count);
        vulkan_status = vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, swapchain_images_.data());
    }

    VkSwapchainCreateInfoKHR SwapChainInfo(VkSurfaceKHR& surface, uint32_t image_count) {
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

    void InitializeSurfaceCapabilities(VkSurfaceKHR& surface) {
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

    VkExtent2D ChooseSwapExtent() {
        if (capabilities_.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities_.currentExtent;
        }
        VkExtent2D actual_extent = { size_.x, size_.y };

        actual_extent.width = ClampNum(actual_extent.width, capabilities_.minImageExtent.width, capabilities_.maxImageExtent.width);
        actual_extent.height = ClampNum(actual_extent.height, capabilities_.minImageExtent.height, capabilities_.maxImageExtent.height);

        return actual_extent;
    }

    void CreateImageViews() {
        swapchain_image_views_.resize(swapchain_images_.size());

        for (uint32_t i = 0; i < swapchain_images_.size(); i++) {
            swapchain_image_views_[i] = ImageHelper::CreateImageView(device_, swapchain_images_[i], surface_format_.format, VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    VkImageView DepthImageView() {

        auto format = ImageHelper::FindDepthFormat(physical_device_);

        ImageHelper::InitializeImage(device_, physical_device_, depth_memory_, depth_image_, size_, format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_TYPE_2D);

        auto depth_image_view = ImageHelper::CreateImageView(device_, depth_image_, format, VK_IMAGE_ASPECT_DEPTH_BIT);

        ProgramLog::OutputLine("Created depth image view.");

        return depth_image_view;
    }

    VkDevice device_;
    VkPhysicalDevice physical_device_;

    VkSurfaceKHR surface_;

    VkQueue queue_;

    uint2 size_;

    VkDeviceMemory depth_memory_;

    VkSurfaceCapabilitiesKHR capabilities_;

    vector<VkSurfaceFormatKHR> formats_;
    vector<VkPresentModeKHR> present_modes_;

    VkResult vulkan_status;
};