#include "gui_driver.hpp"

static inline float f = 0.0f;
static inline int counter = 0;

void VulkanGUIDriver::CreateMainFrame() {
    /*
    *     ImGui::Begin("Welcome!");                          // Create a window called "Hello, world!" and append into it.

    ImGui::Checkbox("Demo Window", &show_demo_window_);      // Edit bools storing our window open/close state
    ImGui::Checkbox("Another Window", &show_another_window_);

    ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    ImGui::ColorEdit3("clear color", (float*) &clear_color_); // Edit 3 floats representing a color

    if (ImGui::Button("Button")) {                           // Buttons return true when clicked (most widgets return true when edited/activated)
        counter++;
    }
    ImGui::SameLine();
    ImGui::Text("counter = %d", counter);

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();
    */

    bool window_exit = false;

    ImGuiWindowFlags gui_flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar;
    ImGui::Begin("Viewport", &window_exit, gui_flags);
    float width_ratio = (screen_width / 10.0f) * 6.0f,
        height_ratio = (screen_height / 10.0f) * 9.0f;
    ImVec2 window_size(width_ratio, height_ratio);
    ImGui::SetWindowSize(window_size);

    ImVec2 window_pos(0.0f, (screen_height - height_ratio) - ((screen_height / 10.0f) * 0.75f));

    ImGui::SetWindowPos(window_pos);
    ImGui::End();
}