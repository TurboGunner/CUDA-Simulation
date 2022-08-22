#pragma once

#include <vulkan/vulkan.h>

#include <stdexcept>
#include <tuple>

using std::tuple;

struct RenderPassInitializer {
	static VkPipelineRasterizationStateCreateInfo RasterizationInfo() {

		VkPipelineRasterizationStateCreateInfo rasterizer {};

		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;

		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;

		rasterizer.lineWidth = 1.0f;
		
		rasterizer.cullMode = VK_CULL_MODE_NONE;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		return rasterizer;
	}

	static VkPipelineMultisampleStateCreateInfo MultiSamplingInfo() {

		VkPipelineMultisampleStateCreateInfo multi_sampling {};

		multi_sampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multi_sampling.sampleShadingEnable = VK_FALSE;
		multi_sampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multi_sampling.minSampleShading = 1.0f;
		multi_sampling.pSampleMask = nullptr;
		multi_sampling.alphaToCoverageEnable = VK_FALSE;
		multi_sampling.alphaToOneEnable = VK_FALSE;

		return multi_sampling;
	}

	static VkPipelineLayoutCreateInfo PipelineLayoutInfo() {
		VkPipelineLayoutCreateInfo pipeline_layout_info {};

		pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipeline_layout_info.setLayoutCount = 0;
		pipeline_layout_info.pSetLayouts = nullptr;

		pipeline_layout_info.pushConstantRangeCount = 0;

		pipeline_layout_info.pPushConstantRanges = nullptr;

		return pipeline_layout_info;
	}

	static tuple<VkSubpassDescription, VkAttachmentDescription> RenderPassDescriptions(const VkFormat& format) {
		VkAttachmentDescription attachment = {};
		attachment.format = format;
		attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference color_attachment_ref = {};
		color_attachment_ref.attachment = 0;
		color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &color_attachment_ref;

		return tuple<VkSubpassDescription, VkAttachmentDescription>(subpass, attachment);
	}

	static VkRenderPassCreateInfo RenderPassInfo(tuple<VkSubpassDescription, VkAttachmentDescription> descriptions) {
		VkSubpassDescription subpass = std::get<0>(descriptions);
		VkAttachmentDescription color_attachment = std::get<1>(descriptions);

		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo render_pass_info = {};
		render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		render_pass_info.attachmentCount = 1;
		render_pass_info.pAttachments = &color_attachment;
		render_pass_info.subpassCount = 1;
		render_pass_info.pSubpasses = &subpass;
		render_pass_info.dependencyCount = 1;
		render_pass_info.pDependencies = &dependency;

		return render_pass_info;
	}
};