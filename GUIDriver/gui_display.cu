#include "gui_driver.cuh"

//#include "../Renderer/raypath.cuh"

//#include "texture_loader.cuh"

#include <tuple>

using std::tuple;

static float f = 0.0f;

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

    float width_ratio = (size_.x / 10.0f) * 6.0f,
        height_ratio = (size_.y / 10.0f) * 9.0f;

    ImVec2 window_size(width_ratio, height_ratio);

    ImGui::SetWindowSize(window_size);
    vulkan_parameters_.ManipulateCamera();

    ImVec2 window_pos(0.0f, (size_.y - height_ratio) - ((size_.y / 10.0f) * 0.75f));

    ImGui::SetWindowPos(window_pos);

    ImGui::End();
}