#include "gui_driver.hpp"

#include "../Renderer/raypath.cuh"

#include "swap_chain_handler.cuh"

static inline float f = 0.0f;
static inline int counter = 0;

void VulkanGUIDriver::CreateMenuBar() {
    std::string menu_name = "Main";
    ImGuiWindowFlags child_flags = ImGuiWindowFlags_MenuBar;

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu(menu_name.c_str())) {
            ImGui::MenuItem("oof");
            ImGui::EndMenu();
        }
    }
    ImGui::EndMainMenuBar();
}

void VulkanGUIDriver::CreateMainFrame() {
    bool window_exit = false;

    ImGuiWindowFlags gui_flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar;
    ImGui::Begin("Viewport", &window_exit, gui_flags);
    float width_ratio = (screen_width / 10.0f) * 6.0f,
        height_ratio = (screen_height / 10.0f) * 9.0f;
    ImVec2 window_size(width_ratio, height_ratio);
    ImGui::SetWindowSize(window_size);

    ImVec2 window_pos(0.0f, (screen_height - height_ratio) - ((screen_height / 10.0f) * 0.75f));

    ImGui::SetWindowPos(window_pos);
    uint2 size;
    size.x = 1024;
    size.y = 1024;

    cudaError_t cuda_status = cudaSuccess;

    RenderImage(size, cuda_status);

    ImGui::End();
}

void VulkanGUIDriver::RenderImage(uint2 size, cudaError_t& cuda_status) {
    VkDeviceSize image_size = size.x * size.y * 4;

    //Vector3D* image_ptr = AllocateTexture(size, cuda_status);
    //vector<Vector3D> image = OutputImage(image_ptr, size);

    ImVec2 uv0 = ImVec2(10.0f / float(size.x), 10.0f / float(size.y));

    SwapChainHandler handler(device_);

    auto tuple = handler.CreateTextureImage(size, cuda_status);

    auto texture = (ImTextureID)ImGui_ImplVulkan_AddTexture(std::get<1>(tuple), std::get<0>(tuple), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    ImVec2 uv1 = ImVec2((10.0f + 1000.0f) / size.x, (10.0f + 1000.0f) / size.y);

    ImGui::Image(texture, ImVec2(size.x, size.y), uv0, uv1);
}