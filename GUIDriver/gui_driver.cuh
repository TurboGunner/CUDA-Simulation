#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include "shader_loader.cuh"
#include "swap_chain_manager.cuh"
#include "cuda_interop_helper.cuh"

#include "mesh_manager.hpp"
#include "sync_structs.hpp"
#include "vertex_data.hpp"
#include "mesh_viewport.hpp"
#include "descriptor_set_handler.hpp"

#include "vulkan_helpers.hpp"
#include "image_helpers.hpp"

#include "vulkan_parameters.hpp"

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

//OpenGL/GLM
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

//Standard Imports
#include <array>
#include <iostream>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

using std::array;
using std::map;
using std::string;
using std::tuple;
using std::vector;

#ifdef _DEBUG
#define IMGUI_VULKAN_DEBUG_REPORT
#endif

static inline vector<string> Split(const string str, const char seperator)
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
    string output = s_stream.str();
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
        VkResult vulkan_status = vkDeviceWaitIdle(device_);
        VulkanErrorHandler(vulkan_status);

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();

        CleanupVulkan();

        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    void VulkanInstantiation();

    VkResult DebugOptionInitialization();

    VkResult PoolDescriptionInitialization();
    VkResult LogicalDeviceInitialization();

    void SelectQueueFamily();

    VkResult DebugErrorCallback();

    VkResult SelectGPU();

    VkResult SetupVulkanWindow();

    void CleanupVulkan();

    void RenderFrame(ImDrawData* draw_data);

    void StartRenderPass(VkCommandBuffer& command_buffer, VkFramebuffer& frame_buffer);

    void EndRenderPass();

    void PresentFrame();

    void LoadInitializationInfo(ImGui_ImplVulkan_InitInfo& init_info);

    VkResult MinimizeRenderCondition(ImDrawData* draw_data, VkCommandBuffer& command_buffer);

    void SwapChainCondition();

    void CreateMenuBar();

    void CreateMainFrame();

    void InitializeVulkan();

    void GUIPollLogic(bool& exit_condition);

    void IMGUIRenderLogic();

    void RunGUI();

    void GUISetup();

    VulkanParameters vulkan_parameters_;

    VkPhysicalDeviceProperties device_properties_;

    VkInstance               instance_ = VK_NULL_HANDLE;
    VkInstanceCreateInfo     instance_info_ = {};
    VkPhysicalDevice         physical_device_ = VK_NULL_HANDLE;
    VkDevice                 device_ = VK_NULL_HANDLE;

    //Queuing
    uint32_t queue_family_ = (uint32_t) - 1;
    VkQueue queue_ = VK_NULL_HANDLE;

    //Descriptor Pool
    VkDescriptorPoolCreateInfo pool_info_ = {};
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    uint32_t min_image_count_ = 2;

    uint32_t set_size_;

    const uint32_t MAX_FRAMES_IN_FLIGHT_ = 2;
    
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSurfaceFormatKHR surface_format_;

    VkPresentModeKHR present_mode_;

    uint2 size_;

    vector<const char*> phys_device_extensions_ = {
        VK_EXT_DEBUG_REPORT_EXTENSION_NAME
    };

    vector<const char*> device_extensions_ = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    //Callbacks
    VkAllocationCallbacks* allocators_ = nullptr;
    VkDebugReportCallbackEXT debug_report_callback_ = VK_NULL_HANDLE;
    VkDebugReportCallbackCreateInfoEXT debug_info_callback_ = {};

    PFN_vkCreateDebugReportCallbackEXT InstanceDebugCallbackEXT;

    //Windows
    SDL_Window* window;

    ImVec4 clear_color_ = ImVec4(0.075f, 0.0875f, 0.1f, 0.332f);

    //IMGUI
    const string program_name = "CUDA CFD Simulator";
};