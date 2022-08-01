#include "gui_driver.hpp"

static inline float f = 0.0f;
static inline int counter = 0;

void VulkanGUIDriver::CreateMainFrame() {
    bool window_exit = false;

    ImGuiWindowFlags gui_flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_MenuBar;
    ImGui::Begin("Viewport", &window_exit, gui_flags);
    float width_ratio = (screen_width / 10.0f) * 6.0f,
        height_ratio = (screen_height / 10.0f) * 9.0f;
    ImVec2 window_size(width_ratio, height_ratio);
    ImGui::SetWindowSize(window_size);

    ImVec2 window_pos(0.0f, (screen_height - height_ratio) - ((screen_height / 10.0f) * 0.75f));

    ImGui::SetWindowPos(window_pos);

    std::string menu_name = "Main";
    ImGuiWindowFlags child_flags = ImGuiWindowFlags_MenuBar;

    float menu_height = width_ratio - (window_pos.y / 2.0f);

    ImVec2 child_size(0.0f, menu_height);

    ImGui::BeginChild(menu_name.c_str(), child_size, false, child_flags);
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu(menu_name.c_str())) {
            ImGui::MenuItem("oof");
            ImGui::EndMenu();
        }
    }
    ImGui::EndMainMenuBar();
    ImGui::EndChild();

    ImGui::End();
}