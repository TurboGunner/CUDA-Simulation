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

    size_.x = 1024;
    size_.y = 1024;

    // Create Window Surface
    VkSurfaceKHR surface;
    if (SDL_Vulkan_CreateSurface(window, instance_, &surface) == 0) {
        ProgramLog::OutputLine("Error: Failed to create Vulkan surface.");
        return;
    }
    ProgramLog::OutputLine("Successfully created Vulkan surface for window!");

    int width, height;
    CreateFrameBuffers(width, height, surface);

    VkFormat format = VK_FORMAT_A8B8G8R8_SRGB_PACK32;

    auto subpass = CreateSubpass(format);

    vulkan_helper_ = VulkanHelper(device_, subpass);
    shader_handler_ = ShaderLoader(device_, subpass, pipeline_cache_, allocators_);

    shader_handler_.CreateGraphicsPipeline();
    vulkan_helper_.CreateCommandPool(command_pool_, queue_family_);

    vulkan_helper_.CreateFrameBuffers();

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
    LoadInitializationInfo(init_info);

    ImGui_ImplVulkan_Init(&init_info, shader_handler_.render_pass_);

    s_stream << "Address for Vulkan Render Pipeline: " << vulkan_helper_.render_pass_ << ".";

    //NOTE!
    //command_buffers_.emplace("GUI", wd_->Frames[wd_->FrameIndex].CommandBuffer);

    texture_handler_ = TextureLoader(device_, physical_device_, command_pool_, queue_family_);

    vulkan_status = vkResetCommandPool(device_, command_pool_, 0);
    VulkanErrorHandler(vulkan_status);

    ProgramLog::OutputLine("Reset IMGUI command pool successfully.");

    //One time commands

    VkCommandBufferBeginInfo begin_info = {};
    BeginRendering(begin_info);

    ProgramLog::OutputLine("Started command recording!");

    VkSubmitInfo end_info = {};
    EndRendering(end_info, command_buffers_["GUI"]);

    vulkan_status = vkEndCommandBuffer(command_buffers_["GUI"]);
    VulkanErrorHandler(vulkan_status);

    ProgramLog::OutputLine("Ended command recording!");

    vulkan_status = vkQueueSubmit(queue_, 1, &end_info, VK_NULL_HANDLE);
    VulkanErrorHandler(vulkan_status);
    
    ProgramLog::OutputLine("\nSuccessfully submitted item to queue!\n");

    VkCommandBufferAllocateInfo info = {};
    VulkanHelper::AllocateCommandBuffer(command_pool_);
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

void VulkanGUIDriver::MinimizeRenderCondition(ImDrawData* draw_data, VkCommandBuffer& command_buffer) {
    const bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);
    if (is_minimized) {
        return;
    }

    clear_values_[0].color.float32[0] = clear_color_.x * clear_color_.w;
    clear_values_[0].color.float32[1] = clear_color_.y * clear_color_.w;
    clear_values_[0].color.float32[2] = clear_color_.z * clear_color_.w;
    clear_values_[0].color.float32[3] = clear_color_.w;

    FrameRender(draw_data, command_buffer);
    FramePresent();
}

void VulkanGUIDriver::StartRenderPass(VkCommandBuffer& command_buffer, VkFramebuffer& frame_buffer) {
    VkRenderPassBeginInfo info = {};

    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    info.renderPass = render_pass_;
    info.framebuffer = frame_buffer;

    info.renderArea.extent.width = size_.x;
    info.renderArea.extent.height = size_.y;

    info.clearValueCount = clear_values_.size();
    info.pClearValues = clear_values_.data();

    vkCmdBeginRenderPass(command_buffer, &info, VK_SUBPASS_CONTENTS_INLINE);
    ProgramLog::OutputLine("Began render pass!");
}

void VulkanGUIDriver::EndRenderPass(VkCommandBuffer& command_buffer, VkSemaphore& image_semaphore, VkSemaphore& render_semaphore) {
    vkCmdEndRenderPass(command_buffer);

    vulkan_status = vkEndCommandBuffer(command_buffer);
    VulkanErrorHandler(vulkan_status);

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo info = {};

    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &image_semaphore;

    info.pWaitDstStageMask = &wait_stage;

    array<VkCommandBuffer, 1> command_buffers =
    { command_buffers_["GUI"] };

    info.commandBufferCount = command_buffers.size();
    info.pCommandBuffers = command_buffers.data();

    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &render_semaphore;

    vulkan_status = vkQueueSubmit(queue_, 1, &info, render_fence_);
    VulkanErrorHandler(vulkan_status);
}

void VulkanGUIDriver::FrameRender(ImDrawData* draw_data, VkCommandBuffer& command_buffer) {
    VkResult vulkan_status;

    uint32_t image_index;
    vulkan_status = vkAcquireNextImageKHR(device_, swap_chain_, UINT64_MAX, image_semaphore_, VK_NULL_HANDLE, &frame_index_);

    if (vulkan_status == VK_ERROR_OUT_OF_DATE_KHR || vulkan_status == VK_SUBOPTIMAL_KHR) {
        swap_chain_rebuilding_ = true;
        return;
    }

    VulkanErrorHandler(vulkan_status);
    vulkan_status = vkWaitForFences(device_, 1, &render_fence_, VK_TRUE, UINT64_MAX); //Has indefinite wait instead of periodic checks
    VulkanErrorHandler(vulkan_status);

    vulkan_status = vkResetFences(device_, 1, &render_fence_);
    VulkanErrorHandler(vulkan_status);

    ManageCommandBuffer(command_pool_, command_buffers_["GUI"]);

    StartRenderPass(command_buffer, vulkan_helper_.frame_buffers_[image_index_]);

    ImGui_ImplVulkan_RenderDrawData(draw_data, command_buffer); // Record imgui primitives into command buffer
    //Note!
    EndRenderPass(command_buffer, image_semaphore_, render_semaphore_);
}

VkRenderPass VulkanGUIDriver::CreateSubpass(VkFormat& format) {
    VkRenderPass subpass{};
    auto render_info = RenderPassInitializer::RenderPassInfo(
        RenderPassInitializer::RenderPassDescriptions(format));

    if (vkCreateRenderPass(device_, &render_info, nullptr, &subpass) != VK_SUCCESS) {
        throw std::runtime_error("Could not create Dear ImGui's render pass");
    }

    return subpass;
}

void VulkanGUIDriver::CleanupVulkanWindow() {
    //NOTE!
}