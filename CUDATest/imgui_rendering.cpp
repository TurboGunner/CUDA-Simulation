#include "gui_driver.hpp"

#include <fstream>

void VulkanGUIDriver::RunGUI() {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0) {
        ProgramLog::OutputLine("SDL_Init failed %s", SDL_GetError());
        return;
    }

    SDL_DisplayMode display_mode;

    if (SDL_GetDesktopDisplayMode(0, &display_mode) != 0) {
        ProgramLog::OutputLine("SDL_GetDesktopDisplayMode failed %s", SDL_GetError());
        return;
    }

    screen_width = display_mode.w;
    screen_height = display_mode.h * 0.95f;

    // Setup window
    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    window = SDL_CreateWindow(program_name.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, screen_width, screen_height, window_flags);

    s_stream << "Created a window with size " << screen_width << " X " << screen_height;
    ProgramLog::OutputLine(s_stream);

    InitializeVulkan();

    // Create Window Surface
    VkSurfaceKHR surface;
    if (SDL_Vulkan_CreateSurface(window, instance_, &surface) == 0) {
        ProgramLog::OutputLine("Error: Failed to create Vulkan surface.");
        return;
    }

    int width, height;
    CreateFrameBuffers(width, height, surface);

    IMGUIRenderLogic();

    vulkan_status = vkDeviceWaitIdle(device_);
    VulkanErrorHandler(vulkan_status);
    ImGui_ImplVulkan_DestroyFontUploadObjects();

    bool exit_condition = false;

    while (!exit_condition) {
        GUIPollLogic(exit_condition);
    }
}

void VulkanGUIDriver::IMGUIRenderLogic() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();

    ImGui_ImplSDL2_InitForVulkan(window);
    ImGui_ImplVulkan_InitInfo init_info = {};
    LoadInitializationInfo(init_info, wd_);

    ImGui_ImplVulkan_Init(&init_info, wd_->RenderPass);
    VkCommandPool command_pool = wd_->Frames[wd_->FrameIndex].CommandPool;
    VkCommandBuffer command_buffer = wd_->Frames[wd_->FrameIndex].CommandBuffer;

    vulkan_status = vkResetCommandPool(device_, command_pool, 0);
    VulkanErrorHandler(vulkan_status);

    VkCommandBufferBeginInfo begin_info = {};
    BeginRendering(begin_info, command_buffer);

    VkSubmitInfo end_info = {};
    EndRendering(end_info, command_buffer);

    vulkan_status = vkEndCommandBuffer(command_buffer);
    VulkanErrorHandler(vulkan_status);

    vulkan_status = vkQueueSubmit(queue_, 1, &end_info, VK_NULL_HANDLE);

    VulkanErrorHandler(vulkan_status);
}

void VulkanGUIDriver::GUIPollLogic(bool& exit_condition) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL2_ProcessEvent(&event);
        if (event.type == SDL_QUIT) {
            exit_condition = true;
        }
        if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(window)) {
            exit_condition = true;
        }
    }
    SwapChainCondition();

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    CreateMenuBar();
    CreateMainFrame();

    // Rendering
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();
    MinimizeRenderCondition(draw_data);
}

void VulkanGUIDriver::InitializeVulkan() {
    uint32_t ext_count = 0;
    SDL_Vulkan_GetInstanceExtensions(window, &ext_count, NULL);

    const char** extensions = new const char* [ext_count];
    SDL_Vulkan_GetInstanceExtensions(window, &ext_count, extensions);

    VulkanInstantiation(extensions, ext_count);
    if (ext_count > 0) {
        delete[] extensions;
    }
}

void VulkanGUIDriver::EndRendering(VkSubmitInfo& end_info, VkCommandBuffer& command_buffer) {
    end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    end_info.commandBufferCount = 1;
    end_info.pCommandBuffers = &command_buffer;
}

void VulkanGUIDriver::MinimizeRenderCondition(ImDrawData* draw_data) {
    const bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);
    if (is_minimized) {
        return;
    }
    wd_->ClearValue.color.float32[0] = clear_color_.x * clear_color_.w;
    wd_->ClearValue.color.float32[1] = clear_color_.y * clear_color_.w;
    wd_->ClearValue.color.float32[2] = clear_color_.z * clear_color_.w;
    wd_->ClearValue.color.float32[3] = clear_color_.w;
    FrameRender(draw_data);
    FramePresent();
}

void VulkanGUIDriver::BeginRendering(VkCommandBufferBeginInfo& begin_info, VkCommandBuffer& command_buffer) {
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vulkan_status = vkBeginCommandBuffer(command_buffer, &begin_info);
    VulkanErrorHandler(vulkan_status);

    ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
}

void VulkanGUIDriver::StartRenderPass(ImGui_ImplVulkanH_Frame* frame_draw) {
    VkRenderPassBeginInfo info = {};

    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    info.renderPass = wd_->RenderPass;
    info.framebuffer = frame_draw->Framebuffer;

    info.renderArea.extent.width = wd_->Width;
    info.renderArea.extent.height = wd_->Height;

    info.clearValueCount = 1;
    info.pClearValues = &wd_->ClearValue;

    vkCmdBeginRenderPass(frame_draw->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
}

void VulkanGUIDriver::EndRenderPass(ImGui_ImplVulkanH_Frame* frame_draw, VkSemaphore image_semaphore, VkSemaphore render_semaphore) {
    vkCmdEndRenderPass(frame_draw->CommandBuffer);
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo info = {};

    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &image_semaphore;

    info.pWaitDstStageMask = &wait_stage;

    info.commandBufferCount = 1;
    info.pCommandBuffers = &frame_draw->CommandBuffer;

    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &render_semaphore;

    vulkan_status = vkEndCommandBuffer(frame_draw->CommandBuffer);
    VulkanErrorHandler(vulkan_status);

    vulkan_status = vkQueueSubmit(queue_, 1, &info, frame_draw->Fence);
    VulkanErrorHandler(vulkan_status);
}

void VulkanGUIDriver::FrameRender(ImDrawData* draw_data) {
    VkResult vulkan_status;

    VkSemaphore image_semaphore = wd_->FrameSemaphores[wd_->SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore render_semaphore = wd_->FrameSemaphores[wd_->SemaphoreIndex].RenderCompleteSemaphore;

    vulkan_status = vkAcquireNextImageKHR(device_, wd_->Swapchain, UINT64_MAX, image_semaphore, VK_NULL_HANDLE, &wd_->FrameIndex);

    if (vulkan_status == VK_ERROR_OUT_OF_DATE_KHR || vulkan_status == VK_SUBOPTIMAL_KHR)
    {
        swap_chain_rebuilding_ = true;
        return;
    }
    VulkanErrorHandler(vulkan_status);

    ImGui_ImplVulkanH_Frame* fd = &wd_->Frames[wd_->FrameIndex];
    vulkan_status = vkWaitForFences(device_, 1, &fd->Fence, VK_TRUE, UINT64_MAX); //Has indefinite wait instead of periodic checks
    VulkanErrorHandler(vulkan_status);

    vulkan_status = vkResetFences(device_, 1, &fd->Fence);
    VulkanErrorHandler(vulkan_status);

    ManageCommandBuffer(fd);

    StartRenderPass(fd);

    ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer); // Record imgui primitives into command buffer
    //Note!
    EndRenderPass(fd, image_semaphore, render_semaphore);
}

void VulkanGUIDriver::CleanupVulkanWindow() {
    ImGui_ImplVulkanH_DestroyWindow(instance_, device_, &main_window_data_, allocators_);
}