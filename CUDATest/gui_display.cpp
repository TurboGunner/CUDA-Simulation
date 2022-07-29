#include "gui_driver.hpp"

void VulkanGUIDriver::CreateFrame() {
    float f = 0.0f;
    int counter = 0;

    ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

    ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
    ImGui::Checkbox("Demo Window", &show_demo_window_);      // Edit bools storing our window open/close state
    ImGui::Checkbox("Another Window", &show_another_window_);

    ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    ImGui::ColorEdit3("clear color", (float*)&clear_color_); // Edit 3 floats representing a color

    if (ImGui::Button("Button")) {                           // Buttons return true when clicked (most widgets return true when edited/activated)
        counter++;
    }
    ImGui::SameLine();
    ImGui::Text("counter = %d", counter);

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();
}