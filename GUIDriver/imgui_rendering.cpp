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

    screen_width = display_mode.w;
    screen_height = display_mode.h * 0.95f;

    // Setup window
    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    window = SDL_CreateWindow(program_name.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, screen_width, screen_height, window_flags);

    s_stream << "Created a window with size " << screen_width << " X " << screen_height << ".";
    ProgramLog::OutputLine(s_stream);

    InitializeVulkan();

    size_.x = screen_width;
    size_.y = screen_height;

    // Create Window Surface
    VkSurfaceKHR surface;
    if (SDL_Vulkan_CreateSurface(window, instance_, &surface) == 0) {
        ProgramLog::OutputLine("Error: Failed to create Vulkan surface.");
        return;
    }

    ProgramLog::OutputLine("Successfully created Vulkan surface for window!");
    surface_ = surface;

    int width, height;
    CreateGUIWindow(width, height, surface);

    bool exit_condition = false;

    GUISetup();

    while (!exit_condition) {
        GUIPollLogic(exit_condition);
    }
}

void VulkanGUIDriver::GUISetup() {
    //Create Swapchain
    swap_chain_helper_ = SwapChainProperties(device_, physical_device_, surface_, queue_, size_);

    swap_chain_ = swap_chain_helper_.Initialize();

    //Setting surface format and present mode
    surface_format_ = swap_chain_helper_.surface_format_;
    present_mode_ = swap_chain_helper_.present_mode_;

    extent_ = swap_chain_helper_.extent_;

    //Create viewport and scissor

    viewport_.x = 0.0f;
    viewport_.y = 0.0f;
    viewport_.width = (float) size_.x;
    viewport_.height = (float) size_.y;
    viewport_.minDepth = 0.0f;
    viewport_.maxDepth = 1.0f;

    scissor_.offset = { 0, 0 };
    scissor_.extent = extent_;

    //Create Renderpass
    render_pass_initializer_ = RenderPassInitializer(device_);
    render_pass_ = render_pass_initializer_.Initialize(surface_format_.format, ImageHelper::FindDepthFormat(physical_device_));

    //Creating all of the other external helpers
    vulkan_helper_ = VulkanHelper(device_, size_, MAX_FRAMES_IN_FLIGHT_);
    shader_handler_ = ShaderLoader(device_, viewport_, scissor_, pipeline_cache_, allocators_);
    sync_struct_ = SyncStruct(device_, MAX_FRAMES_IN_FLIGHT_);
    mesh_data_ = VertexData(device_, physical_device_, queue_, MAX_FRAMES_IN_FLIGHT_);
    mesh_viewport_ = MeshViewport(device_);
    interop_handler_ = CudaInterop(device_, physical_device_);

    //Creating command pool
    vulkan_helper_.CreateCommandPool(command_pool_, queue_family_);

    shader_handler_.InitializeDescriptorHandler(physical_device_, descriptor_pool_, command_pool_, queue_, MAX_FRAMES_IN_FLIGHT_);

    //Creating depth pass for rendering
    swap_chain_helper_.InitializeDepthPass(command_pool_);

    //Creating framebuffers
    ProgramLog::OutputLine("\nSwapchain Image View Size: " + std::to_string(swap_chain_helper_.swapchain_image_views_.size()));

    swap_chain_helper_.CreateSwapchainFrameBuffers(render_pass_, swap_chain_helper_.swapchain_image_views_, swap_chain_helper_.depth_image_view_);

    //Creating command buffers
    if (vulkan_helper_.InitializeCommandBuffers(command_buffers_, command_pool_) != VK_SUCCESS) {
        ProgramLog::OutputLine("Error: Failed to create command buffers!");
        return;
    }

    //Creating synchronization structs (fences and semaphores)
    if (sync_struct_.Initialize() != VK_SUCCESS) {
        ProgramLog::OutputLine("Error: Failed to create synchronization structs!");
        return;
    }

    image_semaphores_ = sync_struct_.present_semaphores_;
    render_semaphores_ = sync_struct_.render_semaphores_;

    render_fences_ = sync_struct_.fences_;

    clear_values_.resize(2);

    clear_values_[0].color = { {0.0f, 0.0f, 0.0f, 1.0f } };
    clear_values_[1].depthStencil = { 1.0f, 0 };

    //Create Pipeline
    shader_handler_.Initialize(render_pass_);

    mesh_data_.Initialize(command_pool_);
    mesh_data_.InitializeIndex(command_pool_);

    mesh_pipeline_layout_ = shader_handler_.mesh_pipeline_layout_;

    IMGUIRenderLogic();

    vulkan_status = vkDeviceWaitIdle(device_);
    VulkanErrorHandler(vulkan_status);

    //Font Renderer
    VkCommandBuffer command_buffer = BeginSingleTimeCommands();
    ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
    EndSingleTimeCommands(command_buffer);
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

    ImGui_ImplVulkan_Init(&init_info, render_pass_);

    s_stream << "\n\nAddress for Vulkan Render Pass: " << render_pass_ << ".\n";
    ProgramLog::OutputLine(s_stream);

    //texture_handler_ = TextureLoader(device_, physical_device_, command_pool_, queue_family_);
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

    MinimizeRenderCondition(draw_data, command_buffers_[current_frame_]);
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

    FrameRender(draw_data);
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

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shader_handler_.render_pipeline_);

    mesh_data_.BindPipeline(command_buffer, command_pool_);

    auto& descriptor_helper = shader_handler_.descriptor_set_handler_;
    //uint32_t uniform_offset = BufferHelpers::PadUniformBufferSize(device_properties_, frame_index_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shader_handler_.mesh_pipeline_layout_, 0, 1, &descriptor_helper.global_descriptors_[current_frame_], 0, nullptr);
}

void VulkanGUIDriver::EndRenderPass(VkCommandBuffer& command_buffer, VkSemaphore& image_semaphore, VkSemaphore& render_semaphore) {
    vkCmdEndRenderPass(command_buffer);

    vulkan_status = vkEndCommandBuffer(command_buffer);
    VulkanErrorHandler(vulkan_status);

    //WIP!
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo info = {};

    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.waitSemaphoreCount = sync_struct_.wait_semaphores_.size();
    info.pWaitSemaphores = sync_struct_.wait_semaphores_.data();

    info.pWaitDstStageMask = sync_struct_.wait_stages_.data();

    info.commandBufferCount = 1;
    info.pCommandBuffers = &command_buffer;

    info.signalSemaphoreCount = sync_struct_.signal_semaphores_.size();
    info.pSignalSemaphores = sync_struct_.signal_semaphores_.data();

    vulkan_status = vkQueueSubmit(queue_, 1, &info, render_fences_[current_frame_]);
    VulkanErrorHandler(vulkan_status);
    //WIP
}

void VulkanGUIDriver::FrameRender(ImDrawData* draw_data) {
    VkResult vulkan_status;

    vulkan_status = vkAcquireNextImageKHR(device_, swap_chain_, UINT64_MAX, image_semaphores_[current_frame_], VK_NULL_HANDLE, &image_index_);

    if (vulkan_status == VK_ERROR_OUT_OF_DATE_KHR || vulkan_status == VK_SUBOPTIMAL_KHR) {
        swap_chain_rebuilding_ = true;
        return;
    }

    VulkanErrorHandler(vulkan_status);
    vulkan_status = vkWaitForFences(device_, 1, &render_fences_[current_frame_], VK_TRUE, UINT64_MAX); //Has indefinite wait instead of periodic checks
    VulkanErrorHandler(vulkan_status);

    vulkan_status = vkResetFences(device_, 1, &render_fences_[current_frame_]);
    VulkanErrorHandler(vulkan_status);

    //WIP, maybe move order after Fence reset if it doesn't work

    //Wait Semaphores (CUDA <-> Vulkan Sync)
    sync_struct_.wait_semaphores_.push_back(image_semaphores_[current_frame_]);
    sync_struct_.wait_stages_.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    sync_struct_.GetWaitSemaphores(frame_index_);

    sync_struct_.GetSignalFrameSemaphores();
    sync_struct_.signal_semaphores_.push_back(render_semaphores_[frame_index_]);

    //WIP

    ManageCommandBuffer(command_pool_, command_buffers_[current_frame_]);

    StartRenderPass(command_buffers_[current_frame_], swap_chain_helper_.frame_buffers_[image_index_]);

    auto constants = mesh_viewport_.ViewportRotation(frame_index_, current_frame_, shader_handler_.descriptor_set_handler_);

    vkCmdPushConstants(command_buffers_[current_frame_], mesh_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants), &constants);

    vkCmdDrawIndexed(command_buffers_[current_frame_], mesh_data_.vertices.Size(), 1, 0, 0, 0);

    ImGui_ImplVulkan_RenderDrawData(draw_data, command_buffers_[current_frame_]);

    vkCmdSetViewport(command_buffers_[current_frame_], 0, 1, &viewport_);
    vkCmdSetScissor(command_buffers_[current_frame_], 0, 1, &scissor_);

    frame_index_++;

    //Note!
    EndRenderPass(command_buffers_[current_frame_], image_semaphores_[current_frame_], render_semaphores_[current_frame_]);
}

VkCommandBuffer VulkanGUIDriver::BeginSingleTimeCommands() {
    return VulkanHelper::BeginSingleTimeCommands(device_, command_pool_);
}

void VulkanGUIDriver::EndSingleTimeCommands(VkCommandBuffer& command_buffer) {
    VulkanHelper::EndSingleTimeCommands(command_buffer, device_, command_pool_, queue_);
}