#include "shader_loader.cuh"

ShaderLoader::ShaderLoader(VkDevice& device_in, VkViewport& viewport_in, VkRect2D& scissor_in, VkPipelineCache & cache_in, VkAllocationCallbacks* allocators) {
    device_ = device_in;

    viewport_ = viewport_in;
    scissor_ = scissor_in;

    pipeline_cache_ = cache_in;

    allocators_ = allocators;
}

void ShaderLoader::InitializeDescriptorHandler(VkPhysicalDevice& phys_device_in, VkDescriptorPool& descriptor_pool_in, VkCommandPool& command_pool_in, VkQueue& queue_in, const size_t & max_frames_const_in) {
    descriptor_set_handler_ = DescriptorSetHandler(device_, phys_device_in, queue_in, command_pool_in, descriptor_pool_in, max_frames_const_in);
}

VkPipelineLayout ShaderLoader::InitializeLayout() {
    auto pipeline_layout_info = PipelineLayoutInfo();

    if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout!");
    }

    return pipeline_layout_;
}


void ShaderLoader::InitializeMeshPipelineLayout() {
    VkPipelineLayoutCreateInfo mesh_pipeline_layout_info = PipelineLayoutInfo();

    VkPushConstantRange push_constant = {};

    push_constant.offset = 0;
    push_constant.size = sizeof(MeshPushConstants);
    push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    mesh_pipeline_layout_info.pPushConstantRanges = &push_constant;
    mesh_pipeline_layout_info.pushConstantRangeCount = 1;

    vulkan_status = descriptor_set_handler_.DescriptorSets();

    array<VkDescriptorSetLayout, 2> set_layouts = { descriptor_set_handler_.global_set_layout_, descriptor_set_handler_.object_set_layout_ };

    descriptor_set_handler_.AllocateDescriptorSets();

    mesh_pipeline_layout_info.setLayoutCount = 1;
    mesh_pipeline_layout_info.pSetLayouts = &set_layouts.data()[0];

    if (vkCreatePipelineLayout(device_, &mesh_pipeline_layout_info, nullptr, &mesh_pipeline_layout_) != VK_SUCCESS) {
        ProgramLog::OutputLine("Error: Could not successfully create the mesh pipeline layout!");
    }
}

VkPipelineLayoutCreateInfo ShaderLoader::PipelineLayoutInfo() {
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};

    pipeline_layout_info.flags = 0;
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 0;
    pipeline_layout_info.pSetLayouts = nullptr;

    pipeline_layout_info.pushConstantRangeCount = 0;
    pipeline_layout_info.pPushConstantRanges = nullptr;

    return pipeline_layout_info;
}

VkPipeline ShaderLoader::Initialize(VkRenderPass & render_pass) {
    auto vert_shader_code = ReadFile("Shaders/shader_vert_test.spv");
    auto frag_shader_code = ReadFile("Shaders/shader_frag_test.spv");

    vert_shader_module_ = CreateShaderModule(vert_shader_code);
    frag_shader_module_ = CreateShaderModule(frag_shader_code);

    vert_shader_stage_info_ = PipelineStageInfo(vert_shader_module_, VK_SHADER_STAGE_VERTEX_BIT);
    frag_shader_stage_info_ = PipelineStageInfo(frag_shader_module_, VK_SHADER_STAGE_FRAGMENT_BIT);

    VertexInputInfo();

    InitializeLayout();
    InitializeMeshPipelineLayout();

    InputAssemblyInfo();

    CreateViewportInfo();

    RasterizationInfo();
    MultiSamplingInfo();

    BlendStates();

    GraphicsPipelineInfo(render_pass);
    array<VkPipelineShaderStageCreateInfo, 2> shader_stages = { vert_shader_stage_info_, frag_shader_stage_info_ };
    pipeline_info_.pStages = shader_stages.data();

    vector<VkVertexInputBindingDescription> binding_description = Vertex::GetBindingDescription();
    vector<VkVertexInputAttributeDescription> attribute_descriptions = Vertex::GetAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertex_input_info = {};

    vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_info.vertexBindingDescriptionCount = binding_description.size();
    vertex_input_info.vertexAttributeDescriptionCount = attribute_descriptions.size();
    vertex_input_info.pVertexBindingDescriptions = binding_description.data();
    vertex_input_info.pVertexAttributeDescriptions = attribute_descriptions.data();
    vertex_input_info.flags = 0;

    pipeline_info_.pVertexInputState = &vertex_input_info;

    s_stream << "Name for shader stage (Index 0): " << pipeline_info_.pStages[0].pName << ".";
    ProgramLog::OutputLine(s_stream);

    pipeline_info_.layout = mesh_pipeline_layout_;

    vulkan_status = vkCreateGraphicsPipelines(device_, pipeline_cache_, 1, &pipeline_info_, VK_NULL_HANDLE, &render_pipeline_);

    ProgramLog::OutputLine("Successfully created graphics pipeline!");

    ProgramLog::OutputLine("Successfully created mesh pipeline!");

    vkDestroyShaderModule(device_, frag_shader_module_, nullptr);
    vkDestroyShaderModule(device_, vert_shader_module_, nullptr);

    return render_pipeline_;
}

void ShaderLoader::Clean() {
    vkDestroyPipeline(device_, render_pipeline_, nullptr);

    vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    vkDestroyPipelineLayout(device_, mesh_pipeline_layout_, nullptr);

    descriptor_set_handler_.Clean();
}