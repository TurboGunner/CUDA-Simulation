#pragma once

#include <vulkan/vulkan.h>

#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>

using std::string;
using std::vector;

class ShaderLoader {
public:
    static vector<char> ReadFile(const string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t)file.tellg();
        vector<char> buffer(fileSize);

        file.close();

        return buffer;
    }

    void CreateGraphicsPipeline() {
        auto vert_shader_code = ReadFile("shaders/vert.spv");
        auto frag_shader_code = ReadFile("shaders/frag.spv");

        VkShaderModule vert_shader_module = CreateShaderModule(vert_shader_code);
        VkShaderModule frag_shader_module = CreateShaderModule(frag_shader_code);

        vkDestroyShaderModule(device_, frag_shader_module, nullptr);
        vkDestroyShaderModule(device_, vert_shader_module, nullptr);

        //auto pipeline_layout_info = PipelineLayoutInfo();

        if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }
    }

    VkShaderModule CreateShaderModule(const vector<char>& code) {
        VkShaderModuleCreateInfo create_info = ShaderModuleInfo(code);
        VkShaderModule shader_module;

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
};