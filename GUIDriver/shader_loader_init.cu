#include "shader_loader.cuh"

void ShaderLoader::RasterizationInfo() {
    rasterization_info_.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterization_info_.depthClampEnable = VK_FALSE;

    rasterization_info_.rasterizerDiscardEnable = VK_FALSE;
    rasterization_info_.polygonMode = VK_POLYGON_MODE_FILL;

    rasterization_info_.lineWidth = 1.0f;

    rasterization_info_.cullMode = VK_CULL_MODE_NONE;
    rasterization_info_.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    rasterization_info_.depthBiasEnable = VK_FALSE;

    rasterization_info_.depthBiasConstantFactor = 0.0f;
    rasterization_info_.depthBiasClamp = 0.0f;
    rasterization_info_.depthBiasSlopeFactor = 0.0f;
}

void ShaderLoader::MultiSamplingInfo() {
    multi_sampling_info_.pNext = nullptr;
    multi_sampling_info_.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multi_sampling_info_.sampleShadingEnable = VK_FALSE;
    multi_sampling_info_.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    multi_sampling_info_.minSampleShading = 1.0f;
    multi_sampling_info_.pSampleMask = nullptr;
    multi_sampling_info_.alphaToCoverageEnable = VK_FALSE;
    multi_sampling_info_.alphaToOneEnable = VK_FALSE;
}

void ShaderLoader::ColorBlendAttachment() {
    color_blend_attachment_.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment_.blendEnable = VK_TRUE;

    color_blend_attachment_.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment_.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

    color_blend_attachment_.colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment_.alphaBlendOp = VK_BLEND_OP_ADD;

    color_blend_attachment_.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    color_blend_attachment_.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

    ProgramLog::OutputLine("Created color blend attachment for the render pipeline.");
}

void ShaderLoader::ColorBlendState() {
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

void ShaderLoader::DepthStencilState(const bool& depth_test, const bool& depth_write, const VkCompareOp& compare_op) {
    depth_stencil_state_.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil_state_.pNext = nullptr;

    depth_stencil_state_.depthTestEnable = depth_test ? VK_TRUE : VK_FALSE;
    depth_stencil_state_.depthWriteEnable = depth_write ? VK_TRUE : VK_FALSE;
    depth_stencil_state_.depthCompareOp = depth_test ? compare_op : VK_COMPARE_OP_ALWAYS;
    depth_stencil_state_.depthBoundsTestEnable = VK_FALSE;
    depth_stencil_state_.minDepthBounds = 0.0f;
    depth_stencil_state_.maxDepthBounds = 1.0f;
    depth_stencil_state_.stencilTestEnable = VK_FALSE;

    ProgramLog::OutputLine("Created depth stencil state for the render pipeline.\n");
}

void ShaderLoader::DynamicStateInfo() {
    //NOTE!
    array<VkDynamicState, 2> dynamic_states = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

    dynamic_state_.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state_.dynamicStateCount = dynamic_states.size();
    dynamic_state_.pDynamicStates = dynamic_states.data();

    ProgramLog::OutputLine("Initialized dynamic states for the render pipeline.\n");
}

void ShaderLoader::CreateViewportInfo() {
    viewport_state_.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state_.pScissors = &scissor_;
    viewport_state_.pViewports = &viewport_;
    viewport_state_.viewportCount = 1;
    viewport_state_.scissorCount = 1;

    ProgramLog::OutputLine("Initialized viewport state values for the render pipeline.");
}

void ShaderLoader::VertexInputInfo() {
    vector<VkVertexInputBindingDescription> binding_description = Vertex::GetBindingDescription();
    vector<VkVertexInputAttributeDescription> attribute_descriptions = Vertex::GetAttributeDescriptions();

    vertex_input_info_.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_info_.vertexBindingDescriptionCount = 0;
    vertex_input_info_.vertexAttributeDescriptionCount = attribute_descriptions.size();
    vertex_input_info_.pVertexBindingDescriptions = nullptr;
    vertex_input_info_.pVertexAttributeDescriptions = attribute_descriptions.data();
    vertex_input_info_.flags = 0;

    ProgramLog::OutputLine("Initialized vertex input info struct for the render pipeline.");
}

void ShaderLoader::InputAssemblyInfo() {
    input_assembly_.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly_.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly_.primitiveRestartEnable = VK_FALSE;
    input_assembly_.pNext = nullptr;

    ProgramLog::OutputLine("Initialized input assembly info struct for the render pipeline.");
}

void ShaderLoader::GraphicsPipelineInfo(VkRenderPass& render_pass) {
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

    ProgramLog::OutputLine("Dynamic State Count: " + std::to_string(dynamic_state_.dynamicStateCount) + ".");

    pipeline_info_.pColorBlendState = &color_blend_state_;
    pipeline_info_.pDynamicState = nullptr;
    pipeline_info_.pNext = nullptr;

    pipeline_info_.layout = pipeline_layout_;
    pipeline_info_.renderPass = render_pass;
    pipeline_info_.subpass = 0;

    pipeline_info_.basePipelineHandle = VK_NULL_HANDLE;

    ProgramLog::OutputLine("Created graphics pipeline info for render pipeline.");
}