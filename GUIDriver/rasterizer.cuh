#pragma once

#include <vulkan/vulkan.h>

//Logging
#include "../CUDATest/handler_classes.hpp"

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

	static tuple<VkSubpassDescription, array<VkAttachmentDescription, 2>> RenderPassDescriptions(VkFormat format) {
		VkAttachmentDescription color_attachment = CreateColorAttachment(format);

		VkAttachmentDescription depth_attachment = CreateDepthAttachment();

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

	static VkAttachmentDescription CreateColorAttachment(VkFormat color_format) {
		VkAttachmentDescription color_attachment = {};

		color_attachment.flags = 0;
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

	static VkAttachmentDescription CreateDepthAttachment() {
		VkAttachmentDescription depth_attachment = {};

		depth_attachment.flags = 0;
		depth_attachment.format = VK_FORMAT_D32_SFLOAT;
		depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		return depth_attachment;
	}

	static VkSubpassDescription CreateSubpassDescription(VkAttachmentReference& color_attachment_ref, VkAttachmentReference& depth_attachment_ref) {
		VkSubpassDescription subpass = {};
		s_stream << "Color Attachment Ref ID: " << color_attachment_ref.layout << "\n";
		ProgramLog::OutputLine(s_stream);

		subpass.flags = 0;
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.inputAttachmentCount = 0;
		subpass.preserveAttachmentCount = 0;
		subpass.pColorAttachments = &color_attachment_ref;
		//subpass.pDepthStencilAttachment = &depth_attachment_ref;
		subpass.pDepthStencilAttachment = nullptr;
		subpass.pResolveAttachments = nullptr;
		subpass.pInputAttachments = nullptr;
		subpass.pPreserveAttachments = nullptr;

		return subpass;
	}

	static VkRenderPassCreateInfo RenderPassInfo(tuple<VkSubpassDescription, array<VkAttachmentDescription, 2>> descriptions, VkDevice& device) {
		VkSubpassDescription subpass = std::get<0>(descriptions);

		//Note!

		VkRenderPassCreateInfo render_pass_info = {};

		auto dependencies = RenderPassDependencies();

		//AAAAAA

		VkAttachmentDescription color_attachment = {};
		color_attachment.format = std::get<1>(descriptions).data()[0].format;
		color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		//AAAAAA

		render_pass_info.flags = 0;
		render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		render_pass_info.attachmentCount = 1;
		//render_pass_info.pAttachments = &std::get<1>(descriptions).data()[0];
		render_pass_info.pAttachments = &color_attachment;
		render_pass_info.subpassCount = 1;
		render_pass_info.pSubpasses = &subpass;
		render_pass_info.dependencyCount = 1;
		//render_pass_info.pDependencies = &dependencies.data()[0];
		render_pass_info.pDependencies = &dependency;

		VkRenderPass render_pass = {};

		if (vkCreateRenderPass(device, &render_pass_info, nullptr, &render_pass) != VK_SUCCESS) {
			throw std::runtime_error("Could not create Dear ImGui's render pass");
		}

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

		return array<VkSubpassDependency, 2> { dependency, depth_dependency };
	}

	static VkRenderPassBeginInfo SubpassBeginInfo(VkRenderPass& subpass, VkFramebuffer& frame_buffer, const uint2& size) {
		VkRenderPassBeginInfo pass_info = {};

		pass_info.renderPass = subpass;
		pass_info.renderArea.offset.x = 0;
		pass_info.renderArea.offset.y = 0;
		pass_info.renderArea.extent.width = size.x;
		pass_info.renderArea.extent.height = size.y;
		pass_info.clearValueCount = 1;
		pass_info.pClearValues = nullptr;
		pass_info.framebuffer = frame_buffer;

		return pass_info;
	}
};