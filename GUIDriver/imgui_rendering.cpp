#include "gui_driver.cuh"
#include "vulkan_helpers.hpp"

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

    size_.x = display_mode.w;
    size_.y = display_mode.h * 0.95f;

    // Setup window
    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    window = SDL_CreateWindow(program_name.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, size_.x, size_.y, window_flags);

    s_stream << "Created a window with size " << size_.x << " X " << size_.y << ".";
    ProgramLog::OutputLine(s_stream);

    InitializeVulkan();

    // Create Window Surface
    if (SDL_Vulkan_CreateSurface(window, instance_, &surface_) == 0) {
        ProgramLog::OutputLine("Error: Failed to create Vulkan surface.");
        return;
    }

    ProgramLog::OutputLine("Successfully created Vulkan surface for window!");

    //int width = 0, height = 0;

    //SDL_GetWindowSize(window, &width, &height);
    SetupVulkanWindow();

    bool exit_condition = false;

    GUISetup();

    while (!exit_condition) {
        GUIPollLogic(exit_condition);
    }
}

void VulkanGUIDriver::GUISetup() {
    vulkan_parameters_ = VulkanParameters(device_, physical_device_, surface_, MAX_FRAMES_IN_FLIGHT_, queue_, queue_family_, descriptor_pool_, size_);

    vulkan_parameters_.InitVulkan();
    IMGUIRenderLogic();

    VkResult vulkan_status = vkDeviceWaitIdle(device_);

    //Font Renderer
    VkCommandBuffer command_buffer = vulkan_parameters_.BeginSingleTimeCommands();

    ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
    vulkan_parameters_.EndSingleTimeCommands(command_buffer);

    ImGui_ImplVulkan_DestroyFontUploadObjects();
}

void VulkanGUIDriver::IMGUIRenderLogic() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForVulkan(window);
    ImGui_ImplVulkan_InitInfo init_info = {};
    LoadInitializationInfo(init_info);

    ImGui_ImplVulkan_Init(&init_info, vulkan_parameters_.render_pass_);

    s_stream << "\n\nAddress for Vulkan Render Pass: " << vulkan_parameters_.render_pass_ << ".\n";
    ProgramLog::OutputLine(s_stream);
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
    ImGui_ImplSDL2_NewFrame(window);
    ImGui::NewFrame();

    CreateMenuBar();
    CreateMainFrame();

    // Rendering
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();

    VkCommandBuffer in_flight_command_buffer = vulkan_parameters_.InFlightCommandBuffer();
    MinimizeRenderCondition(draw_data, in_flight_command_buffer);
}

VkResult VulkanGUIDriver::MinimizeRenderCondition(ImDrawData* draw_data, VkCommandBuffer& command_buffer) {
    VkResult vulkan_status = VK_SUCCESS;
    const bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);
    if (is_minimized) {
        return vulkan_status;
    }

    vulkan_parameters_.clear_values_[0].color.float32[0] = clear_color_.x * clear_color_.w;
    vulkan_parameters_.clear_values_[0].color.float32[1] = clear_color_.y * clear_color_.w;
    vulkan_parameters_.clear_values_[0].color.float32[2] = clear_color_.z * clear_color_.w;
    vulkan_parameters_.clear_values_[0].color.float32[3] = clear_color_.w;

    RenderFrame(draw_data);
    PresentFrame();

    cudaError_t cuda_status = vulkan_parameters_.InteropDrawFrame();
    vulkan_status = vkDeviceWaitIdle(device_);
    return vulkan_status;
}

void VulkanGUIDriver::StartRenderPass(VkCommandBuffer& command_buffer, VkFramebuffer& frame_buffer) {
    VkRenderPassBeginInfo info = {};

    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;

    info.renderPass = vulkan_parameters_.render_pass_;
    info.framebuffer = frame_buffer;

    info.renderArea.extent.width = size_.x;
    info.renderArea.extent.height = size_.y;

    info.clearValueCount = vulkan_parameters_.clear_values_.size();
    info.pClearValues = vulkan_parameters_.clear_values_.data();

    vkCmdBeginRenderPass(command_buffer, &info, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vulkan_parameters_.render_pipeline_);

    //uint32_t uniform_offset = BufferHelpers::PadUniformBufferSize(device_properties_, frame_index_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vulkan_parameters_.mesh_pipeline_layout_, 0, 1, &vulkan_parameters_.CurrentDescriptorSet(), 0, nullptr);

    vulkan_parameters_.BindMeshPipeline();
}

void VulkanGUIDriver::EndRenderPass() {
    VkCommandBuffer in_flight_command_buffer = vulkan_parameters_.InFlightCommandBuffer();
    vkCmdEndRenderPass(in_flight_command_buffer);

    VkResult vulkan_status = vkEndCommandBuffer(in_flight_command_buffer);
    VulkanErrorHandler(vulkan_status);

    //WIP!
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo info = {};

    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.waitSemaphoreCount = vulkan_parameters_.WaitSemaphores().size();
    info.pWaitSemaphores = vulkan_parameters_.WaitSemaphores().data();

    info.pWaitDstStageMask = vulkan_parameters_.WaitStages().data();

    info.commandBufferCount = 1;
    info.pCommandBuffers = &in_flight_command_buffer;

    info.signalSemaphoreCount = vulkan_parameters_.SignalSemaphores().size();
    info.pSignalSemaphores = vulkan_parameters_.SignalSemaphores().data();

    vulkan_status = vkQueueSubmit(queue_, 1, &info, vulkan_parameters_.InFlightFence());
    VulkanErrorHandler(vulkan_status);
    //WIP
}

void VulkanGUIDriver::RenderFrame(ImDrawData* draw_data) {
    VkSemaphore in_flight_image_semaphore = vulkan_parameters_.InFlightImageSemaphore();
    VkSemaphore in_flight_render_semaphore = vulkan_parameters_.InFlightRenderSemaphore();

    VkFence in_flight_fence = vulkan_parameters_.InFlightFence();

    VkFramebuffer current_swapchain_buffer = vulkan_parameters_.CurrentSwapchainFrameBuffer();

    vulkan_parameters_.sync_struct_.ClearInteropSynchronization();

    VkResult vulkan_status = vkAcquireNextImageKHR(device_, vulkan_parameters_.swap_chain_, UINT64_MAX, in_flight_image_semaphore, VK_NULL_HANDLE, &vulkan_parameters_.image_index_);

    if (vulkan_status == VK_ERROR_OUT_OF_DATE_KHR || vulkan_status == VK_SUBOPTIMAL_KHR) {
        vulkan_status = vulkan_parameters_.RebuildSwapchain();
        return;
    }

    //WIP, maybe move order after Fence reset if it doesn't work

    //Wait Semaphores (CUDA <-> Vulkan Sync)
    vulkan_parameters_.sync_struct_.wait_semaphores_.push_back(in_flight_image_semaphore);
    vulkan_parameters_.sync_struct_.wait_stages_.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    vulkan_parameters_.sync_struct_.GetWaitSemaphores(vulkan_parameters_.frame_index_);

    vulkan_parameters_.sync_struct_.GetSignalFrameSemaphores();
    vulkan_parameters_.sync_struct_.signal_semaphores_.push_back(in_flight_render_semaphore);

    //WIP

    VulkanErrorHandler(vulkan_status);
    vulkan_status = vkWaitForFences(device_, 1, &in_flight_fence, VK_TRUE, UINT64_MAX); //Has indefinite wait instead of periodic checks
    VulkanErrorHandler(vulkan_status);

    vulkan_status = vkResetFences(device_, 1, &in_flight_fence);
    VulkanErrorHandler(vulkan_status);

    VkCommandBuffer in_flight_command_buffer = vulkan_parameters_.InFlightCommandBuffer();

    VkCommandBufferBeginInfo command_buffer_begin_info = {};
    command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    command_buffer_begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vulkan_status = vkBeginCommandBuffer(in_flight_command_buffer, &command_buffer_begin_info);
    VulkanErrorHandler(vulkan_status);

    StartRenderPass(in_flight_command_buffer, current_swapchain_buffer);

    auto constants = vulkan_parameters_.ViewportRotation();

    vkCmdPushConstants(in_flight_command_buffer, vulkan_parameters_.mesh_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants), &constants);

    //vkCmdDrawIndexed(in_flight_command_buffer, vulkan_parameters_.mesh_data_.vertices.Size(), 1, 0, 0, 0);

    vulkan_parameters_.DrawVerticesCall(); //WIP

    ImGui_ImplVulkan_RenderDrawData(draw_data, in_flight_command_buffer);

    vkCmdSetViewport(in_flight_command_buffer, 0, 1, &vulkan_parameters_.viewport_);
    vkCmdSetScissor(in_flight_command_buffer, 0, 1, &vulkan_parameters_.scissor_);

    vulkan_parameters_.frame_index_++;

    //Note!
    EndRenderPass();
}