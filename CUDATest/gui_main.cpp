#include "gui_driver.hpp"

VulkanGUIDriver::~VulkanGUIDriver() {
    vulkan_status = vkDeviceWaitIdle(device_);
    VulkanErrorHandler(vulkan_status);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    CleanupVulkanWindow();
    CleanupVulkan();

    SDL_DestroyWindow(window);
    SDL_Quit();
}