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

	RenderPassInitializer(VkDevice& device_in);

	VkRenderPass Initialize(const VkFormat& color_format, const VkFormat& depth_format);

	void Clean();

	VkRenderPass render_pass_;

private:
	void RenderPassDescriptions(const VkFormat& format, const VkFormat& depth_format, const bool& depth);

	void CreateColorAttachment(const VkFormat& color_format);

	void CreateDepthAttachment(const VkFormat& depth_format);

	void CreateSubpassDescription(const bool& depth);

	void CreateRenderPass(VkRenderPass& render_pass, const bool& depth = true);

	void RenderPassDependencies();

	VkDevice device_;

	VkSubpassDescription subpass_info_ = {};

	VkRenderPassCreateInfo render_pass_info_ = {};

	VkAttachmentDescription color_attachment_ = {}, depth_attachment_ = {};

	VkSubpassDependency dependency_ = {}, depth_dependency_ = {};
	VkAttachmentReference color_attachment_ref_ = {}, depth_attachment_ref_ = {};

	bool init_status_ = false;
};