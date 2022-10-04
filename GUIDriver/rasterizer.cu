#include "rasterizer.cuh"

RenderPassInitializer::RenderPassInitializer(VkDevice& device_in) {
	device_ = device_in;
}

VkRenderPass RenderPassInitializer::Initialize(const VkFormat& color_format, const VkFormat& depth_format) {
	RenderPassDescriptions(color_format, depth_format, true);

	CreateRenderPass(render_pass_);

	ProgramLog::OutputLine("Successfully initialized the render pass!\n\n");

	return render_pass_;
}

void RenderPassInitializer::Clean() {
	vkDestroyRenderPass(device_, render_pass_, nullptr);
}

void RenderPassInitializer::RenderPassDescriptions(const VkFormat& format, const VkFormat& depth_format, const bool& depth) {
	if (!init_status_) {
		CreateColorAttachment(format);
		CreateDepthAttachment(depth_format);

		color_attachment_ref_.attachment = 0;
		color_attachment_ref_.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		depth_attachment_ref_.attachment = 1;
		depth_attachment_ref_.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		ProgramLog::OutputLine("Created attachment references for the render pass.");

		init_status_ = true;
	}
	CreateSubpassDescription(depth);
}

void RenderPassInitializer::CreateColorAttachment(const VkFormat& color_format) {
	color_attachment_.format = color_format;
	color_attachment_.samples = VK_SAMPLE_COUNT_1_BIT;
	color_attachment_.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	color_attachment_.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	color_attachment_.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment_.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment_.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	color_attachment_.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	ProgramLog::OutputLine("Created color attachment for the render pass.");
}

void RenderPassInitializer::CreateDepthAttachment(const VkFormat& depth_format) {
	depth_attachment_.format = depth_format;
	depth_attachment_.samples = VK_SAMPLE_COUNT_1_BIT;
	depth_attachment_.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment_.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depth_attachment_.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment_.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depth_attachment_.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depth_attachment_.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	ProgramLog::OutputLine("Created depth attachment for the render pass.");
}

void RenderPassInitializer::CreateSubpassDescription(const bool& depth) {
	s_stream << "\n\nColor Attachment Ref ID: " << color_attachment_ref_.layout << "\n";
	ProgramLog::OutputLine(s_stream);

	subpass_info_.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass_info_.colorAttachmentCount = 1;
	subpass_info_.inputAttachmentCount = 0;
	subpass_info_.preserveAttachmentCount = 0;
	subpass_info_.pColorAttachments = &color_attachment_ref_;
	subpass_info_.pDepthStencilAttachment = depth ? &depth_attachment_ref_ : nullptr;
	subpass_info_.pResolveAttachments = nullptr;
	subpass_info_.pInputAttachments = nullptr;
	subpass_info_.pPreserveAttachments = nullptr;

	ProgramLog::OutputLine("Created subpass description for the render pass.");
}

void RenderPassInitializer::CreateRenderPass(VkRenderPass& render_pass, const bool& depth) {
	RenderPassDependencies();

	array<VkSubpassDependency, 2> dependencies = { dependency_, depth_dependency_ };
	array<VkAttachmentDescription, 2> attachments = { color_attachment_, depth_attachment_ };

	render_pass_info_.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_info_.attachmentCount = depth ? attachments.size() : attachments.size() - 1;
	render_pass_info_.pAttachments = depth ? attachments.data() : &attachments.data()[0];
	render_pass_info_.subpassCount = 1;
	render_pass_info_.pSubpasses = &subpass_info_;
	render_pass_info_.dependencyCount = depth ? dependencies.size() : dependencies.size() - 1;
	render_pass_info_.pDependencies = depth ? dependencies.data() : &dependencies.data()[0];

	ProgramLog::OutputLine("Initialized the renderpass creation info.");

	if (vkCreateRenderPass(device_, &render_pass_info_, nullptr, &render_pass) != VK_SUCCESS) {
		throw std::runtime_error("Could not create Dear ImGui's render pass");
	}
}

void RenderPassInitializer::RenderPassDependencies() {
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