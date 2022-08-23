#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <stdexcept>
#include <tuple>

using std::array;
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

	static tuple<VkSubpassDescription, array<VkAttachmentDescription, 2>> RenderPassDescriptions(const VkFormat& format) {
		VkAttachmentDescription color_attachment = CreateColorAttachment(format);

		VkAttachmentDescription depth_attachment = CreateDepthAttachment(VK_FORMAT_D32_SFLOAT);

		VkAttachmentReference color_attachment_ref = {};
		color_attachment_ref.attachment = 0;
		color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depth_attachment_ref = {};
		depth_attachment_ref.attachment = 1;
		depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = CreateSubpassDescription(color_attachment_ref, depth_attachment_ref);

		array<VkAttachmentDescription, 2> attachments = { color_attachment, depth_attachment };

		return tuple<VkSubpassDescription, array<VkAttachmentDescription, 2>>(subpass, attachments);
	}

	static VkAttachmentDescription CreateColorAttachment(const VkFormat& color_format) {
		VkAttachmentDescription color_attachment = {};

		//color_attachment.flags = 0;
		color_attachment.format = color_format;
		color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		return color_attachment;
	}

	static VkAttachmentDescription CreateDepthAttachment(const VkFormat& depth_format) {
		VkAttachmentDescription depth_attachment = {};

		depth_attachment.flags = 0;
		depth_attachment.format = depth_format;
		depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		return depth_attachment;
	}

	static VkSubpassDescription CreateSubpassDescription(VkAttachmentReference& color_attachment_ref, VkAttachmentReference& depth_attachment_ref) {
		VkSubpassDescription subpass = {};

		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.inputAttachmentCount = 0;
		subpass.preserveAttachmentCount = 0;
		subpass.pColorAttachments = &color_attachment_ref;
		subpass.pDepthStencilAttachment = &depth_attachment_ref;
		subpass.pPreserveAttachments = nullptr;
		subpass.pInputAttachments = nullptr;
		subpass.pResolveAttachments = nullptr;

		return subpass;
	}

	static VkRenderPassCreateInfo RenderPassInfo(tuple<VkSubpassDescription, array<VkAttachmentDescription, 2>> descriptions) {
		VkSubpassDescription subpass = std::get<0>(descriptions);

		array<VkSubpassDependency, 2> dependencies = RenderPassDependencies();

		VkRenderPassCreateInfo render_pass_info = {};
		render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		render_pass_info.attachmentCount = 2;
		render_pass_info.pAttachments = std::get<1>(descriptions).data();
		render_pass_info.subpassCount = 1;
		render_pass_info.pSubpasses = &subpass;
		render_pass_info.dependencyCount = 2;
		render_pass_info.pDependencies = dependencies.data();

		return render_pass_info;
	}

	static array<VkSubpassDependency, 2> RenderPassDependencies() {
		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkSubpassDependency depth_dependency = {};
		depth_dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		depth_dependency.dstSubpass = 0;
		depth_dependency.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		depth_dependency.srcAccessMask = 0;
		depth_dependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		depth_dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		return { dependency, depth_dependency };
	}
};