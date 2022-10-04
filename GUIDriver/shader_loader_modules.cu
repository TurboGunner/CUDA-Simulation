#include "shader_loader.cuh"

vector<uint32_t> ShaderLoader::ReadFile(const string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file " + filename + "!");
    }

    size_t file_size = (size_t)file.tellg();
    vector<uint32_t> buffer(file_size / sizeof(uint32_t));

    file.seekg(0);
    file.read((char*)buffer.data(), file_size);

    file.close();

    ProgramLog::OutputLine("Successfully read shader file " + filename + "!\n");

    return buffer;
}

VkShaderModule ShaderLoader::CreateShaderModule(const vector<uint32_t>& code) {
    VkShaderModuleCreateInfo create_info = ShaderModuleInfo(code);

    VkShaderModule shader_module = {};

    if (vkCreateShaderModule(device_, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
        ProgramLog::OutputLine("Failed to create shader module!");
    }
    return shader_module;
}

VkShaderModuleCreateInfo ShaderLoader::ShaderModuleInfo(const vector<uint32_t>& code) {
    VkShaderModuleCreateInfo create_info = {};

    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = code.size() * sizeof(uint32_t);
    create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

    return create_info;
}

VkPipelineShaderStageCreateInfo ShaderLoader::PipelineStageInfo(VkShaderModule& shader_module, const VkShaderStageFlagBits& flag) {
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