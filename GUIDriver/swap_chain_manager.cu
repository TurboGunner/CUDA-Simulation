#include "swap_chain_manager.cuh"

SwapChainProperties::SwapChainProperties(VkDevice& device_in, VkPhysicalDevice& phys_device_in, VkSurfaceKHR& surface_in, VkQueue& queue_in, uint2& size_in) {
    device_ = device_in;
    physical_device_ = phys_device_in;
    surface_ = surface_in;
    queue_ = queue_in;

    size_ = size_in;
}

VkSwapchainKHR SwapChainProperties::Initialize() {
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

void SwapChainProperties::InitializeDepthPass(VkCommandPool& command_pool) {
    depth_image_view_ = DepthImageView();

    ImageHelper::TransitionImageLayout(device_, command_pool, queue_, depth_image_, surface_format_.format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
}

void SwapChainProperties::CreateSwapchainFrameBuffers(VkRenderPass& render_pass, vector<VkImageView>& swapchain_image_views, VkImageView& depth_image_view) {
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

        VkResult vulkan_status = vkCreateFramebuffer(device_, &frame_buffer_info, nullptr, &frame_buffers_[i]);
    }
}

VkResult SwapChainProperties::RecreateSwapChain(VkRenderPass& render_pass, VkCommandPool& command_pool) {
    VkResult vulkan_status = vkDeviceWaitIdle(device_);

    Clean();

    Initialize();
    CreateImageViews();
    CreateSwapchainFrameBuffers(render_pass, swapchain_image_views_, depth_image_view_);
    InitializeDepthPass(command_pool);

    return vulkan_status;
}

void SwapChainProperties::Clean() {
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