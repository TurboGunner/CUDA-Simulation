#pragma once

#include "vertex_data.hpp"
#include "mesh_manager.hpp"
#include "descriptor_set_handler.hpp"

//#include "vulkan_parameters.hpp"

#include "imgui.h"
#include "imgui_impl_vulkan.h"

#include <vulkan/vulkan.h>

#include <string>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <array>

using std::string;
using std::tuple;
using std::array;
using std::vector;

class ShaderLoader {
public:
    ShaderLoader() = default;

    ShaderLoader(VkDevice& device_in, VkViewport& viewport_in, VkRect2D& scissor_in, VkPipelineCache& cache_in, VkAllocationCallbacks* allocators = nullptr);

    void InitializeDescriptorHandler(VkPhysicalDevice& phys_device_in, VkDescriptorPool& descriptor_pool_in, VkCommandPool& command_pool_in, VkQueue& queue_in, const size_t& max_frames_const_in);

    VkPipelineLayout InitializeLayout();

    void InitializeMeshPipelineLayout();

    static VkPipelineLayoutCreateInfo PipelineLayoutInfo();

    VkPipeline Initialize(VkRenderPass& render_pass);

    void Clean();

    //Helper Struct
    DescriptorSetHandler descriptor_set_handler_;

    VkPipelineLayout pipeline_layout_, mesh_pipeline_layout_;
    VkPipeline render_pipeline_;

private:
    vector<uint32_t> ReadFile(const string& filename);

    VkShaderModule CreateShaderModule(const vector<uint32_t>& code);

    VkShaderModuleCreateInfo ShaderModuleInfo(const vector<uint32_t>& code);

    VkPipelineShaderStageCreateInfo PipelineStageInfo(VkShaderModule& shader_module, const VkShaderStageFlagBits& flag);

    void BlendStates() {
        DynamicStateInfo();

        ColorBlendAttachment();

        ColorBlendState();
        DepthStencilState();
    }

    void RasterizationInfo();

    void MultiSamplingInfo();

    void ColorBlendAttachment();

    void ColorBlendState();

    void DepthStencilState(const bool& depth_test = true, const bool& depth_write = true, const VkCompareOp& compare_op = VK_COMPARE_OP_LESS_OR_EQUAL);

    void DynamicStateInfo();

    void CreateViewportInfo();

    void VertexInputInfo();

    void InputAssemblyInfo();

    void GraphicsPipelineInfo(VkRenderPass& render_pass);

    VkDevice device_;
    VkPipelineCache pipeline_cache_;

    VkPipelineDynamicStateCreateInfo dynamic_state_ = {};

    VkPipelineColorBlendAttachmentState color_blend_attachment_ = {};
    VkPipelineColorBlendStateCreateInfo color_blend_state_ = {};

    VkPipelineDepthStencilStateCreateInfo depth_stencil_state_ = {};

    VkPipelineVertexInputStateCreateInfo vertex_input_info_ = {};

    VkPipelineInputAssemblyStateCreateInfo input_assembly_ = {};

    VkPipelineViewportStateCreateInfo viewport_state_ = {};

    VkPipelineRasterizationStateCreateInfo rasterization_info_ = {};
    VkPipelineMultisampleStateCreateInfo multi_sampling_info_ = {};

    VkViewport viewport_ = {};
    VkRect2D scissor_ = {};

    VkPipelineShaderStageCreateInfo vert_shader_stage_info_ = {}, frag_shader_stage_info_ = {};

    VkGraphicsPipelineCreateInfo pipeline_info_ = {};

    VkShaderModule vert_shader_module_ = {}, frag_shader_module_ = {};

    VkResult vulkan_status;

    VkAllocationCallbacks* allocators_;
};