#pragma once

#include "rasterizer.cuh"
#include "gui_driver.cuh"

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

    ShaderLoader(VkDevice device_in, ImGui_ImplVulkanH_Window* wd_in, VkPipelineCache cache_in) {
        device_ = device_in;
        wd_ = wd_in;

        pipeline_cache_ = cache_in;
    }

    static vector<char> ReadFile(const string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t) file.tellg();
        vector<char> buffer(fileSize);

        file.close();

        return buffer;
    }

    tuple<VkPipelineColorBlendStateCreateInfo, VkPipelineDynamicStateCreateInfo> BlendStates() {
        array<VkDynamicState, 2> dynamic_states = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamic_state {};
        dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
        dynamic_state.pDynamicStates = dynamic_states.data();

        VkPipelineColorBlendAttachmentState color_blend_attachment {};
        color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        color_blend_attachment.blendEnable = VK_FALSE;
        color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
        color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo color_blending{};
        color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blending.logicOpEnable = VK_FALSE;
        color_blending.logicOp = VK_LOGIC_OP_COPY; 
        color_blending.attachmentCount = 1;
        color_blending.pAttachments = &color_blend_attachment;
        color_blending.blendConstants[0] = 0.0f; 
        color_blending.blendConstants[1] = 0.0f; 
        color_blending.blendConstants[2] = 0.0f; 
        color_blending.blendConstants[3] = 0.0f; 

        return tuple<VkPipelineColorBlendStateCreateInfo, VkPipelineDynamicStateCreateInfo>(color_blending, dynamic_state);
    }

    void CreateGraphicsPipeline() {
        auto vert_shader_code = ReadFile("Shaders/vert.spv");
        auto frag_shader_code = ReadFile("Shaders/frag.spv");

        VkShaderModule vert_shader_module = CreateShaderModule(vert_shader_code);
        VkShaderModule frag_shader_module = CreateShaderModule(frag_shader_code);

        auto pipeline_layout_info = RenderPassInitializer::PipelineLayoutInfo();

        if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        auto vert_shader_stage_info = PipelineStageInfo(vert_shader_module, VK_SHADER_STAGE_VERTEX_BIT);
        auto frag_shader_stage_info = PipelineStageInfo(frag_shader_module, VK_SHADER_STAGE_FRAGMENT_BIT);

        array<VkPipelineShaderStageCreateInfo, 2> shader_stages = { vert_shader_stage_info, frag_shader_stage_info };

        VkPipelineVertexInputStateCreateInfo vertex_input_info {};

        VkPipelineInputAssemblyStateCreateInfo input_assembly {};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly.primitiveRestartEnable = VK_FALSE;

        vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input_info.vertexBindingDescriptionCount = 0;
        vertex_input_info.vertexAttributeDescriptionCount = 0;

        VkPipelineViewportStateCreateInfo viewport_state {};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.scissorCount = 1;

        auto rasterization_info = RenderPassInitializer::RasterizationInfo();
        auto multi_sampling_info = RenderPassInitializer::MultiSamplingInfo();

        auto blend_states = BlendStates();

        VkGraphicsPipelineCreateInfo pipeline_info {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.stageCount = 2;
        pipeline_info.pStages = shader_stages.data();
        pipeline_info.pVertexInputState = &vertex_input_info;
        pipeline_info.pInputAssemblyState = &input_assembly;
        pipeline_info.pViewportState = &viewport_state;
        pipeline_info.pRasterizationState = &rasterization_info;
        pipeline_info.pMultisampleState = &multi_sampling_info;
        pipeline_info.pColorBlendState = &std::get<0>(blend_states);
        pipeline_info.pDynamicState = &std::get<1>(blend_states);
        pipeline_info.layout = pipeline_layout_;
        pipeline_info.renderPass = wd_->RenderPass;
        pipeline_info.subpass = 0;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

        vulkan_status = vkCreateGraphicsPipelines(device_, pipeline_cache_, 1, &pipeline_info, VK_NULL_HANDLE, &render_pipeline_);
    }

    VkPipelineShaderStageCreateInfo PipelineStageInfo(VkShaderModule shader_module, VkShaderStageFlagBits flag) {
        VkPipelineShaderStageCreateInfo shader_stage_info{};
        shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stage_info.stage = flag;

        shader_stage_info.module = shader_module;
        shader_stage_info.pName = "main";
        return shader_stage_info;
    }

    VkShaderModule CreateShaderModule(const vector<char>& code) {
        VkShaderModuleCreateInfo create_info = ShaderModuleInfo(code);
        VkShaderModule shader_module {};

        if (vkCreateShaderModule(device_, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }
        return shader_module;
    }

    VkPipelineShaderStageCreateInfo ShaderStageInfo(VkShaderModule& shader_module) {
        VkPipelineShaderStageCreateInfo frag_shader_stage_info {};

        frag_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        frag_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        frag_shader_stage_info.module = shader_module;
        frag_shader_stage_info.pName = "main";
    }

    VkShaderModuleCreateInfo ShaderModuleInfo(const vector<char>& code) {
        VkShaderModuleCreateInfo create_info{};

        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = code.size();
        create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

        return create_info;
    }

    VkDevice device_;
    VkPipelineCache pipeline_cache_;
    VkPipelineLayout pipeline_layout_;

    VkPipeline render_pipeline_;

    ImGui_ImplVulkanH_Window* wd_;

    VkResult vulkan_status;
};