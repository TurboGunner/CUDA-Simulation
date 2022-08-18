#include "gui_driver.cuh"
#include "rasterizer.cuh"

#include <array>
#include <fstream>

using std::array;

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

    s_stream << "Created a window with size " << screen_width << " X " << screen_height << ".";
    ProgramLog::OutputLine(s_stream);

    InitializeVulkan();

    // Create Window Surface
    VkSurfaceKHR surface;
    if (SDL_Vulkan_CreateSurface(window, instance_, &surface) == 0) {
        ProgramLog::OutputLine("Error: Failed to create Vulkan surface.");
        return;
    }
    ProgramLog::OutputLine("Successfully created Vulkan surface for window!");

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
    ImGuiIO& io = ImGui::GetIO();

    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForVulkan(window);
    ImGui_ImplVulkan_InitInfo init_info = {};
    LoadInitializationInfo(init_info, wd_);

    ImGui_ImplVulkan_Init(&init_info, Draw(&wd_->Frames[wd_->FrameIndex]));

    s_stream << "Address for Vulkan Render Pipeline: " << wd_->Pipeline << ".";

    command_pool_ = wd_->Frames[wd_->FrameIndex].CommandPool;
    command_buffers_.emplace("GUI", wd_->Frames[wd_->FrameIndex].CommandBuffer);

    texture_handler_ = TextureLoader(device_, physical_device_, command_pool_, queue_family_);

    vulkan_status = vkResetCommandPool(device_, command_pool_, 0);
    VulkanErrorHandler(vulkan_status);

    ProgramLog::OutputLine("Reset IMGUI command pool successfully.");

    VkCommandBufferBeginInfo begin_info = {};
    BeginRendering(begin_info);

    Draw(&wd_->Frames[wd_->FrameIndex]);

    ProgramLog::OutputLine("Started command recording!");

    VkSubmitInfo end_info = {};
    EndRendering(end_info, command_buffers_["GUI"]);

    vulkan_status = vkEndCommandBuffer(command_buffers_["GUI"]);
    VulkanErrorHandler(vulkan_status);

    ProgramLog::OutputLine("Ended command recording!");

    vulkan_status = vkQueueSubmit(queue_, 1, &end_info, VK_NULL_HANDLE);
    VulkanErrorHandler(vulkan_status);
    
    ProgramLog::OutputLine("\nSuccessfully submitted item to queue!\n");
    //AAA

    VkCommandBufferAllocateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.pNext = nullptr;

    info.commandPool = command_pool_;
    info.commandBufferCount = 1;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    vkAllocateCommandBuffers(device_, &info, &texture_handler_.command_buffer);
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

    uint2 size;
    size.x = 1024;
    size.y = 1024;

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

void VulkanGUIDriver::BeginRendering(VkCommandBufferBeginInfo& begin_info) {
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vulkan_status = vkBeginCommandBuffer(command_buffers_["GUI"], &begin_info);
    VulkanErrorHandler(vulkan_status);

    ImGui_ImplVulkan_CreateFontsTexture(command_buffers_["GUI"]);
    ProgramLog::OutputLine("Started commands buffers.");
}

void VulkanGUIDriver::EndRendering(VkSubmitInfo& end_info, VkCommandBuffer& command_buffer) {
    end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    end_info.commandBufferCount = 1;
    end_info.pCommandBuffers = &command_buffer;
    ProgramLog::OutputLine("Ended command buffers.");
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
    ProgramLog::OutputLine("Began render pass!");
}

void VulkanGUIDriver::EndRenderPass(ImGui_ImplVulkanH_Frame* frame_draw, VkSemaphore image_semaphore, VkSemaphore render_semaphore) {
    vkCmdEndRenderPass(frame_draw->CommandBuffer);

    vulkan_status = vkEndCommandBuffer(frame_draw->CommandBuffer);
    VulkanErrorHandler(vulkan_status);

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo info = {};

    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &image_semaphore;

    info.pWaitDstStageMask = &wait_stage;

    array<VkCommandBuffer, 2> command_buffers =
    { command_buffers_["GUI"], texture_handler_.command_buffer };

    info.commandBufferCount = command_buffers.size();
    info.pCommandBuffers = command_buffers.data();

    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &render_semaphore;

    vulkan_status = vkQueueSubmit(queue_, 1, &info, frame_draw->Fence);
    VulkanErrorHandler(vulkan_status);
}

void VulkanGUIDriver::FrameRender(ImDrawData* draw_data) {
    VkResult vulkan_status;

    VkSemaphore image_semaphore = wd_->FrameSemaphores[wd_->SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore render_semaphore = wd_->FrameSemaphores[wd_->SemaphoreIndex].RenderCompleteSemaphore;

    vulkan_status = vkAcquireNextImageKHR(device_, wd_->Swapchain, UINT64_MAX, image_semaphore, VK_NULL_HANDLE, &wd_->FrameIndex);

    if (vulkan_status == VK_ERROR_OUT_OF_DATE_KHR || vulkan_status == VK_SUBOPTIMAL_KHR) {
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

VkRenderPass VulkanGUIDriver::Draw(ImGui_ImplVulkanH_Frame* frame_draw) {
    VkSubmitInfo end_info = {};
    EndRendering(end_info, texture_handler_.command_buffer);

    VkRenderPass subpass{};
    auto render_info = RenderPassInitializer::RenderPassInfo(
        RenderPassInitializer::RenderPassDescriptions(wd_->SurfaceFormat.format));

    if (vkCreateRenderPass(device_, &render_info, nullptr, &subpass) != VK_SUCCESS) {
        throw std::runtime_error("Could not create Dear ImGui's render pass");
    }

    vulkan_helper_ = VulkanHelper(device_, subpass);

    shader_handler_ = ShaderLoader(device_, wd_, subpass, pipeline_cache_, allocators_);

    VkRenderPassBeginInfo pass_info = {};
    pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    pass_info.pNext = nullptr;

    pass_info.renderPass = subpass;
    pass_info.renderArea.offset.x = 0;
    pass_info.renderArea.offset.y = 0;
    pass_info.renderArea.extent.width = wd_->Width;
    pass_info.renderArea.extent.height = wd_->Height;
    pass_info.clearValueCount = 1;
    pass_info.pClearValues = nullptr;
    pass_info.framebuffer = vulkan_helper_.frame_buffers_[wd_->FrameIndex];

    return subpass;

    //shader_handler_.CreateGraphicsPipeline();
    vkCmdBeginRenderPass(texture_handler_.command_buffer, &pass_info, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(texture_handler_.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shader_handler_.render_pipeline_);
    vkCmdDraw(texture_handler_.command_buffer, 3, 1, 1, 0);

    vkCmdEndRenderPass(texture_handler_.command_buffer);

    vulkan_status = vkEndCommandBuffer(texture_handler_.command_buffer);
    VulkanErrorHandler(vulkan_status);

    vulkan_status = vkQueueSubmit(queue_, 1, &end_info, VK_NULL_HANDLE);
    VulkanErrorHandler(vulkan_status);

    vkEndCommandBuffer(texture_handler_.command_buffer);
    vkResetCommandBuffer(texture_handler_.command_buffer, 0);
}

void VulkanGUIDriver::CleanupVulkanWindow() {
    ImGui_ImplVulkanH_DestroyWindow(instance_, device_, &main_window_data_, allocators_);
}