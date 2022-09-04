#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "shader_loader.cuh"
#include "texture_loader.cuh"
#include "swap_chain_manager.cuh"
#include "mesh_manager.hpp"
#include "sync_structs.hpp"
#include "vertex_data.hpp"

#include "vulkan_helpers.hpp"
#include "image_helpers.hpp"

//Logging
#include "../CUDATest/handler_classes.hpp"

//Graphics Libraries

//IMGUI
#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_vulkan.h"

#include <stdio.h>

//SDL
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

//Vulkan
#include <vulkan/vulkan.h>

//Standard Imports
#include <array>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>
#include <tuple>
#include <map>

using std::array;
using std::string;
using std::vector;
using std::tuple;
using std::map;

#ifdef _DEBUG
#define IMGUI_VULKAN_DEBUG_REPORT
#endif

static inline vector<string> Split(string str, char seperator)
{
    vector<string> strings;
    int currIndex = 0, i = 0;
    int startIndex = 0, endIndex = 0;
    while (i <= str.length()) {
        if (str[i] == seperator || i == str.length()) {
            endIndex = i;
            string subStr = "";
            subStr.append(str, startIndex, endIndex - startIndex);
            strings.push_back(subStr);
            currIndex += 1;
            startIndex = endIndex + 1;
        }
        i++;
    }
    return strings;
}

static void VulkanErrorHandler(VkResult vulkan_status) {
    if (vulkan_status != 0) {
        ProgramLog::OutputLine("[vulkan] Error: VkResult = %d\n", vulkan_status);
    }
}

#ifdef IMGUI_VULKAN_DEBUG_REPORT
VKAPI_ATTR static VkBool32 VKAPI_CALL DebugReport(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType, uint64_t object, size_t location, int32_t messageCode, const char* pLayerPrefix, const char* pMessage, void* pUserData) {
    (void)flags; (void)object; (void)location; (void)messageCode; (void)pUserData; (void)pLayerPrefix; // Unused arguments
    string message = pMessage;
    auto parts = Split(message, '|');

    s_stream << "[vulkan] Debug report from ObjectType: " << objectType << "\nMessage: \n\n";
    for (const string& part : parts) {
        s_stream << part << "\n";
    }
    ProgramLog::OutputLine(s_stream);
    return VK_FALSE;
}
#endif

class VulkanGUIDriver {
public:
    //Default Constructor
    VulkanGUIDriver() = default;

    //Destructor
    ~VulkanGUIDriver() {
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

    void LoadInstanceProperties(const char** extensions, const uint32_t& ext_count);

    void VulkanInstantiation(const char** extensions, uint32_t ext_count);

    void DebugOptionInitialization(const char** extensions, const uint32_t& ext_count);

    void PoolDescriptionInitialization();
    void LoadPoolDescriptionProperties(VkDescriptorPoolCreateInfo& pool_info_, VkDescriptorPoolSize pool_sizes[]);
    void LogicalDeviceInitialization();

    void SelectQueueFamily();

    void DebugErrorCallback();

    void SelectGPU();

    void SetupVulkanWindow(VkSurfaceKHR& surface, int width, int height);

    void CleanupVulkan();

    void CleanupVulkanWindow();

    void FrameRender(ImDrawData* draw_data, VkCommandBuffer& command_buffer);

    void ManageCommandBuffer(VkCommandPool& command_pool, VkCommandBuffer& command_buffer);

    void StartRenderPass(VkCommandBuffer& command_buffer, VkFramebuffer& frame_buffer);

    void EndRenderPass(VkCommandBuffer& command_buffer, VkSemaphore& image_semaphore, VkSemaphore& render_semaphore);

    void FramePresent();

    void LoadInitializationInfo(ImGui_ImplVulkan_InitInfo& init_info);

    void CreateWindow(int& width, int& height, VkSurfaceKHR& surface);

    void MinimizeRenderCondition(ImDrawData* draw_data, VkCommandBuffer& command_buffer);

    VkCommandBuffer BeginSingleTimeCommands();

    void EndSingleTimeCommands(VkCommandBuffer& command_buffer);

    void SwapChainCondition();

    void CreateMenuBar();

    void CreateMainFrame();

    void InitializeVulkan();

    void GUIPollLogic(bool& exit_condition);

    void IMGUIRenderLogic();

    void RunGUI();

    void GUISetup();

    TextureLoader texture_handler_;
    ShaderLoader shader_handler_;
    VulkanHelper vulkan_helper_;
    SwapChainProperties swap_chain_helper_;
    SyncStruct sync_struct_;
    RenderPassInitializer render_pass_initializer_;
    VertexData mesh_data_;

    tuple<VkImageView, VkSampler> image_alloc_;

    VkInstance               instance_ = VK_NULL_HANDLE;
    VkInstanceCreateInfo     instance_info_ = {};
    VkPhysicalDevice         physical_device_ = VK_NULL_HANDLE;
    VkDevice                 device_ = VK_NULL_HANDLE;

    //Queuing
    uint32_t queue_family_ = (uint32_t)-1;
    VkQueue                  queue_ = VK_NULL_HANDLE;

    VkPipelineCache          pipeline_cache_ = VK_NULL_HANDLE;

    //Descriptor Pool
    VkDescriptorPoolCreateInfo pool_info_ = {};
    VkDescriptorPool         descriptor_pool_ = VK_NULL_HANDLE;

    uint32_t                 min_image_count_ = 3;
    bool                     swap_chain_rebuilding_ = false;

    VkViewport viewport_;
    VkRect2D scissor_;

    VkExtent2D extent_;

    VkRenderPass render_pass_;

    VkCommandPool command_pool_;
    vector<VkCommandBuffer> command_buffers_;

    VkSwapchainKHR swap_chain_;

    uint32_t image_index_, current_frame_, frame_index_;

    const uint32_t MAX_FRAMES_IN_FLIGHT_ = 2;
    
    VkSurfaceKHR surface_;
    VkSurfaceFormatKHR surface_format_;

    VkPresentModeKHR present_mode_;

    uint2 size_;

    //Callbacks
    VkAllocationCallbacks* allocators_ = nullptr;
    VkDebugReportCallbackEXT debug_report_callback_ = VK_NULL_HANDLE;
    VkDebugReportCallbackCreateInfoEXT debug_info_callback_ = {};

    PFN_vkCreateDebugReportCallbackEXT InstanceDebugCallbackEXT;

    VkSemaphore image_semaphore_, render_semaphore_;
    VkFence render_fence_;

    vector<VkClearValue> clear_values_;

    //Windows
    SDL_Window* window;

    ImVec4 clear_color_ = ImVec4(0.075f, 0.0875f, 0.1f, 1.00f);

    //Internal Booleans
    bool show_demo_window_, show_another_window_;

    //Error Management
    VkResult vulkan_status;

    //IMGUI
    const string program_name = "CUDA CFD Simulator";
    float screen_width, screen_height;
};