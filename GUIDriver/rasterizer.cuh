#pragma once

#include "vulkan_parameters.hpp"

//Logging
#include "../CUDATest/handler_classes.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <stdexcept>
#include <tuple>
#include <functional>

using std::array;
using std::tuple;
using std::reference_wrapper;

class RenderPassInitializer {
public:
	RenderPassInitializer() = default;

	RenderPassInitializer(VkDevice& device_in) {
		device_ = device_in;
	}

	VkRenderPass Initialize(VkFormat color_format) {
		RenderPassDescriptions(color_format);
		CreateRenderPass();

		ProgramLog::OutputLine("Successfully initialized the renderpass!");

		return render_pass_;
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

		ProgramLog::OutputLine("Created the subpass begin command info.");

		return pass_info;
	}

	VkRenderPass render_pass_;

private:
	void RenderPassDescriptions(VkFormat format) {
		CreateColorAttachment(format);
		CreateDepthAttachment();

		color_attachment_ref_.attachment = 0;
		color_attachment_ref_.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		depth_attachment_ref_.attachment = 1;
		depth_attachment_ref_.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		ProgramLog::OutputLine("Created attachment references for the renderpass.");

		CreateSubpassDescription();
	}

	void CreateColorAttachment(VkFormat color_format) {
		color_attachment_.format = color_format;
		color_attachment_.samples = VK_SAMPLE_COUNT_1_BIT;
		color_attachment_.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		color_attachment_.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		color_attachment_.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		color_attachment_.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		color_attachment_.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		color_attachment_.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		ProgramLog::OutputLine("Created color attachment for renderpass.");
	}

	void CreateDepthAttachment() {
		depth_attachment_.format = VK_FORMAT_D32_SFLOAT;
		depth_attachment_.samples = VK_SAMPLE_COUNT_1_BIT;
		depth_attachment_.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depth_attachment_.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		depth_attachment_.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depth_attachment_.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depth_attachment_.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depth_attachment_.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		ProgramLog::OutputLine("Created depth attachment for renderpass.");
	}

	void CreateSubpassDescription() {
		s_stream << "Color Attachment Ref ID: " << color_attachment_ref_.layout << "\n";
		ProgramLog::OutputLine(s_stream);

		subpass_info_.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass_info_.colorAttachmentCount = 1;
		subpass_info_.inputAttachmentCount = 0;
		subpass_info_.preserveAttachmentCount = 0;
		subpass_info_.pColorAttachments = &color_attachment_ref_;
		subpass_info_.pDepthStencilAttachment = &depth_attachment_ref_;
		subpass_info_.pResolveAttachments = nullptr;
		subpass_info_.pInputAttachments = nullptr;
		subpass_info_.pPreserveAttachments = nullptr;

		ProgramLog::OutputLine("Created subpass description for renderpass.");
	}

	void CreateRenderPass() {
		RenderPassDependencies();

		array<VkSubpassDependency, 2> dependencies = {dependency_, depth_dependency_};
		array<VkAttachmentDescription, 2> attachments = { color_attachment_, depth_attachment_ };

		render_pass_info_.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		render_pass_info_.attachmentCount = attachments.size();
		render_pass_info_.pAttachments = attachments.data();
		render_pass_info_.subpassCount = 1;
		render_pass_info_.pSubpasses = &subpass_info_;
		render_pass_info_.dependencyCount = dependencies.size();
		render_pass_info_.pDependencies = dependencies.data();

		ProgramLog::OutputLine("Initialized the renderpass creation info.");

		if (vkCreateRenderPass(device_, &render_pass_info_, nullptr, &render_pass_) != VK_SUCCESS) {
			throw std::runtime_error("Could not create Dear ImGui's render pass");
		}
	}

	void RenderPassDependencies() {
		dependency_.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency_.dstSubpass = 0;
		dependency_.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency_.srcAccessMask = 0;
		dependency_.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency_.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		depth_dependency_.srcSubpass = VK_SUBPASS_EXTERNAL;
		depth_dependency_.dstSubpass = 0;
		depth_dependency_.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		depth_dependency_.srcAccessMask = 0;
		depth_dependency_.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		depth_dependency_.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		ProgramLog::OutputLine("Created the subpass dependencies for the renderpass.");
	}

	VkDevice device_;

	VkSubpassDescription subpass_info_ = {};

	VkRenderPassCreateInfo render_pass_info_ = {};

	VkAttachmentDescription color_attachment_ = {}, depth_attachment_ = {};

	VkSubpassDependency dependency_ = {}, depth_dependency_ = {};
	VkAttachmentReference color_attachment_ref_ = {}, depth_attachment_ref_ = {};
};