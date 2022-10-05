#pragma once

#define NOMINMAX

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "vulkan_helpers.hpp"
#include "image_helpers.hpp"

#include "../CUDATest/handler_classes.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <limits>
#include <tuple>
#include <vector>

using std::array;
using std::tuple;
using std::vector;

class SwapChainProperties {
public:
    SwapChainProperties() = default;

    SwapChainProperties(VkDevice& device_in, VkPhysicalDevice& phys_device_in, VkSurfaceKHR& surface_in, VkQueue& queue_in, uint2& size_in);

    static uint32_t ClampNum(const uint32_t& value, const uint32_t& min, const uint32_t& max);

    VkSwapchainKHR Initialize();

    void InitializeDepthPass(VkCommandPool& command_pool);

    void CreateSwapchainFrameBuffers(VkRenderPass& render_pass, vector<VkImageView>& swapchain_image_views, VkImageView& depth_image_view);

    void RecreateSwapChain(VkRenderPass& render_pass, VkCommandPool& command_pool);

    void Clean();

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
    void AllocateImages(uint32_t image_count);

    VkSwapchainCreateInfoKHR SwapChainInfo(VkSurfaceKHR& surface, uint32_t image_count);

    void InitializeSurfaceCapabilities(VkSurfaceKHR& surface);

    VkSurfaceFormatKHR ChooseSwapSurfaceFormat();

    VkPresentModeKHR ChooseSwapPresentMode();

    VkExtent2D ChooseSwapExtent();

    void CreateImageViews();

    VkImageView DepthImageView();

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