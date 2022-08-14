#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "shader_loader.cuh"
#include "texture_loader.cuh"

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
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>
#include <tuple>

using std::string;
using std::vector;
using std::tuple;

#ifdef _DEBUG
#define IMGUI_VULKAN_DEBUG_REPORT
#endif

static void VulkanErrorHandler(VkResult vulkan_status) {
    if (vulkan_status == 0)
        return;
    fprintf(stderr, "[vulkan] Error: VkResult = %d\n", vulkan_status);
}

#ifdef IMGUI_VULKAN_DEBUG_REPORT
VKAPI_ATTR static VkBool32 VKAPI_CALL DebugReport(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType, uint64_t object, size_t location, int32_t messageCode, const char* pLayerPrefix, const char* pMessage, void* pUserData) {
    (void)flags; (void)object; (void)location; (void)messageCode; (void)pUserData; (void)pLayerPrefix; // Unused arguments
    fprintf(stderr, "[vulkan] Debug report from ObjectType: %i\nMessage: %s\n\n", objectType, pMessage);
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

    void SetupVulkanWindow(VkSurfaceKHR surface, int width, int height);

    void CleanupVulkan();

    void CleanupVulkanWindow();

    void FrameRender(ImDrawData* draw_data);

    void ManageCommandBuffer(ImGui_ImplVulkanH_Frame* frame_draw);

    void StartRenderPass(ImGui_ImplVulkanH_Frame* frame_draw);

    void EndRenderPass(ImGui_ImplVulkanH_Frame* frame_draw, VkSemaphore image_semaphore, VkSemaphore render_semaphore);

    void FramePresent();

    void LoadInitializationInfo(ImGui_ImplVulkan_InitInfo& init_info, ImGui_ImplVulkanH_Window* window);

    void CreateFrameBuffers(int& width, int& height, VkSurfaceKHR& surface);

    void BeginRendering(VkCommandBufferBeginInfo& begin_info);

    void EndRendering(VkSubmitInfo& end_info, VkCommandBuffer& command_buffer);

    void MinimizeRenderCondition(ImDrawData* draw_data);

    void SwapChainCondition();

    void CreateMenuBar();

    void CreateMainFrame();

    void RenderCall(uint2 size, cudaError_t& cuda_status);
    void RenderImage(uint2 size, cudaError_t& cuda_status);

    void InitializeVulkan();

    void GUIPollLogic(bool& exit_condition);

    void IMGUIRenderLogic();

    void RunGUI();

    TextureLoader texture_handler_;
    ShaderLoader shader_handler_;

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

    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    
    vector<VkImageView> swap_chain_image_views_;
    vector<VkFramebuffer> swap_chain_frame_buffers_;

    //Callbacks
    VkAllocationCallbacks* allocators_ = nullptr;
    VkDebugReportCallbackEXT debug_report_callback_ = VK_NULL_HANDLE;
    VkDebugReportCallbackCreateInfoEXT debug_info_callback_ = {};

    PFN_vkCreateDebugReportCallbackEXT InstanceDebugCallbackEXT;

    //Windows
    SDL_Window* window;
    ImGui_ImplVulkanH_Window* wd_;

    ImGui_ImplVulkanH_Window main_window_data_;

    ImVec4 clear_color_ = ImVec4(0.075f, 0.0875f, 0.1f, 1.00f);

    //Internal Booleans
    bool show_demo_window_, show_another_window_;

    //Error Management
    VkResult vulkan_status;

    //IMGUI
    const string program_name = "CUDA CFD Simulator";
    float screen_width, screen_height;
};