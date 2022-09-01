#pragma once

#include "rasterizer.cuh"
#include "gui_driver.cuh"
#include "vulkan_parameters.hpp"

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

    ShaderLoader(VkDevice& device_in, VkViewport& viewport_in, VkRect2D& scissor_in, VkPipelineCache& cache_in, VkAllocationCallbacks* allocators = nullptr) {
        device_ = device_in;

        viewport_ = viewport_in;
        scissor_ = scissor_in;

        pipeline_cache_ = cache_in;

        allocators_ = allocators;
    }

    VkPipelineLayout InitializeLayout() {
        auto pipeline_layout_info = PipelineLayoutInfo();

        if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout!");
        }

        return pipeline_layout_;
    }

    VkPipeline Initialize(VkRenderPass& render_pass) {
        auto vert_shader_code = ReadFile("Shaders/vert.spv");
        auto frag_shader_code = ReadFile("Shaders/frag.spv");

        vert_shader_module_ = CreateShaderModule(vert_shader_code);
        frag_shader_module_ = CreateShaderModule(frag_shader_code);

        vert_shader_stage_info_ = PipelineStageInfo(vert_shader_module_, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_stage_info_ = PipelineStageInfo(frag_shader_module_, VK_SHADER_STAGE_FRAGMENT_BIT);

        InitializeLayout();

        VertexInputInfo();

        InputAssemblyInfo();

        CreateViewportInfo();

        RasterizationInfo();
        MultiSamplingInfo();

        BlendStates();

        GraphicsPipelineInfo(render_pass);
        array<VkPipelineShaderStageCreateInfo, 2> shader_stages = { vert_shader_stage_info_, frag_shader_stage_info_ };
        pipeline_info_.pStages = shader_stages.data();

        s_stream << "Name for shader stage (Index 0): " << pipeline_info_.pStages[0].pName << ".";
        ProgramLog::OutputLine(s_stream);

        vulkan_status = vkCreateGraphicsPipelines(device_, pipeline_cache_, 1, &pipeline_info_, VK_NULL_HANDLE, &render_pipeline_);

        ProgramLog::OutputLine("Successfully created graphics pipeline!");

        vkDestroyShaderModule(device_, frag_shader_module_, nullptr);
        vkDestroyShaderModule(device_, vert_shader_module_, nullptr);

        return render_pipeline_;
    }

    VkPipelineLayout pipeline_layout_ = {};
    VkPipeline render_pipeline_ = {};

private:
    void RasterizationInfo() {
        rasterization_info_.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterization_info_.depthClampEnable = VK_FALSE;

        rasterization_info_.rasterizerDiscardEnable = VK_FALSE;
        rasterization_info_.polygonMode = VK_POLYGON_MODE_FILL;

        rasterization_info_.lineWidth = 1.0f;

        rasterization_info_.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterization_info_.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        rasterization_info_.depthBiasEnable = VK_FALSE;

        rasterization_info_.depthBiasConstantFactor = 0.0f;
        rasterization_info_.depthBiasClamp = 0.0f;
        rasterization_info_.depthBiasSlopeFactor = 0.0f;
    }

    void MultiSamplingInfo() {
        multi_sampling_info_.pNext = nullptr;
        multi_sampling_info_.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multi_sampling_info_.sampleShadingEnable = VK_FALSE;
        multi_sampling_info_.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        multi_sampling_info_.minSampleShading = 1.0f;
        multi_sampling_info_.pSampleMask = nullptr;
        multi_sampling_info_.alphaToCoverageEnable = VK_FALSE;
        multi_sampling_info_.alphaToOneEnable = VK_FALSE;
    }

    VkPipelineLayoutCreateInfo PipelineLayoutInfo() {
        VkPipelineLayoutCreateInfo pipeline_layout_info = {};

        pipeline_layout_info.flags = 0;
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 0;
        pipeline_layout_info.pSetLayouts = nullptr;

        pipeline_layout_info.pushConstantRangeCount = 0;
        pipeline_layout_info.pPushConstantRanges = nullptr;

        return pipeline_layout_info;
    }

    vector<uint32_t> ReadFile(const string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file " + filename + "!");
        }

        size_t file_size = (size_t) file.tellg();
        vector<uint32_t> buffer(file_size / sizeof(uint32_t));

        file.seekg(0);
        file.read((char*) buffer.data(), file_size);

        file.close();

        ProgramLog::OutputLine("Successfully read shader file " + filename + "!\n");

        return buffer;
    }

     void BlendStates() {
        DynamicStateInfo();

        ColorBlendAttachment();

        ColorBlendState();
        DepthStencilState();
     }

     void ColorBlendAttachment() {
         color_blend_attachment_.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
         color_blend_attachment_.blendEnable = VK_TRUE;

         color_blend_attachment_.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
         color_blend_attachment_.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;

         color_blend_attachment_.colorBlendOp = VK_BLEND_OP_ADD;
         color_blend_attachment_.alphaBlendOp = VK_BLEND_OP_ADD;

         color_blend_attachment_.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
         color_blend_attachment_.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;

         ProgramLog::OutputLine("Created color blend attachment for the render pipeline.");
     }

     void ColorBlendState() {
         color_blend_state_.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
         color_blend_state_.logicOpEnable = VK_FALSE;
         color_blend_state_.logicOp = VK_LOGIC_OP_COPY;
         color_blend_state_.attachmentCount = 1;
         color_blend_state_.pAttachments = &color_blend_attachment_;

         color_blend_state_.blendConstants[0] = 0.0f;
         color_blend_state_.blendConstants[1] = 0.0f;
         color_blend_state_.blendConstants[2] = 0.0f;
         color_blend_state_.blendConstants[3] = 0.0f;

         ProgramLog::OutputLine("Created color blend state for the render pipeline.");
     }

     void DepthStencilState() {
         depth_stencil_state_.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
         depth_stencil_state_.pNext = nullptr;

         depth_stencil_state_.depthTestEnable = VK_TRUE;
         depth_stencil_state_.depthWriteEnable = VK_TRUE;
         depth_stencil_state_.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
         depth_stencil_state_.depthBoundsTestEnable = VK_FALSE;
         depth_stencil_state_.minDepthBounds = 0.0f;
         depth_stencil_state_.maxDepthBounds = 1.0f;
         depth_stencil_state_.stencilTestEnable = VK_FALSE;

         ProgramLog::OutputLine("Created depth stencil state for the render pipeline.\n");
     }

    void DynamicStateInfo() {
        //NOTE!
        array<VkDynamicState, 2> dynamic_states = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

        dynamic_state_.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic_state_.dynamicStateCount = dynamic_states.size();
        dynamic_state_.pDynamicStates = dynamic_states.data();

        ProgramLog::OutputLine("Initialized dynamic states for the render pipeline.\n");
    }

    void CreateViewportInfo() {
        viewport_state_.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state_.pScissors = &scissor_;
        viewport_state_.pViewports = &viewport_;
        viewport_state_.viewportCount = 1;
        viewport_state_.scissorCount = 1;

        ProgramLog::OutputLine("Initialized viewport state values for the render pipeline.");
    }

    void GraphicsPipelineInfo(VkRenderPass& render_pass) {
        array<VkPipelineShaderStageCreateInfo, 2> shader_stages = { vert_shader_stage_info_, frag_shader_stage_info_ };

        pipeline_info_.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

        pipeline_info_.stageCount = shader_stages.size();
        pipeline_info_.pStages = shader_stages.data();
        pipeline_info_.pVertexInputState = &vertex_input_info_;

        pipeline_info_.pInputAssemblyState = &input_assembly_;
        pipeline_info_.pViewportState = &viewport_state_;

        pipeline_info_.pRasterizationState = &rasterization_info_;
        pipeline_info_.pMultisampleState = &multi_sampling_info_;
        pipeline_info_.pDepthStencilState = &depth_stencil_state_;

        pipeline_info_.pColorBlendState = &color_blend_state_;
        //pipeline_info_.pDynamicState = &dynamic_state_;
        pipeline_info_.pDynamicState = nullptr;
        pipeline_info_.pNext = nullptr;

        pipeline_info_.layout = pipeline_layout_;
        pipeline_info_.renderPass = render_pass;
        pipeline_info_.subpass = 0;

        pipeline_info_.basePipelineHandle = VK_NULL_HANDLE;

        ProgramLog::OutputLine("Created graphics pipeline info for render pipeline.");
    }

    void VertexInputInfo() {
        vertex_input_info_.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input_info_.vertexBindingDescriptionCount = 0;
        vertex_input_info_.vertexAttributeDescriptionCount = 0;
        vertex_input_info_.pNext = VK_NULL_HANDLE;
        vertex_input_info_.pVertexBindingDescriptions = nullptr;
        vertex_input_info_.pVertexAttributeDescriptions = nullptr;

        ProgramLog::OutputLine("Initialized vertex input info struct for the render pipeline.");
    }

    void InputAssemblyInfo() {
        input_assembly_.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly_.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly_.primitiveRestartEnable = VK_FALSE;
        input_assembly_.pNext = nullptr;

        ProgramLog::OutputLine("Initialized input assembly info struct for the render pipeline.");
    }

    VkPipelineShaderStageCreateInfo PipelineStageInfo(VkShaderModule& shader_module, const VkShaderStageFlagBits& flag) {
        VkPipelineShaderStageCreateInfo shader_stage_info = {};

        shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stage_info.stage = flag;
        shader_stage_info.module = shader_module;
        shader_stage_info.pName = "main";
        shader_stage_info.pNext = nullptr;
        shader_stage_info.pSpecializationInfo = nullptr;
        shader_stage_info.flags = 0;

        string mode = flag == VK_SHADER_STAGE_VERTEX_BIT ? "vertex" : "frag";
        ProgramLog::OutputLine("Initialized input assembly info struct (" + mode + ") for the render pipeline.");

        return shader_stage_info;
    }

    VkShaderModule CreateShaderModule(const vector<uint32_t>& code) {
        VkShaderModuleCreateInfo create_info = ShaderModuleInfo(code);

        VkShaderModule shader_module = {};

        if (vkCreateShaderModule(device_, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
            ProgramLog::OutputLine("Failed to create shader module!");
        }
        return shader_module;
    }

    VkShaderModuleCreateInfo ShaderModuleInfo(const vector<uint32_t>& code) {
        VkShaderModuleCreateInfo create_info = {};

        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = code.size() * sizeof(uint32_t);
        create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

        return create_info;
    }

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