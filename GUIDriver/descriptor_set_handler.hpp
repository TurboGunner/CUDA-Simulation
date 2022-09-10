#pragma once

#include <vulkan/vulkan.h>

class DescriptorSetHandler {
public:
	DescriptorSetHandler() = default;
	
	DescriptorSetHandler(VkDevice& device_in) {
		device_ = device_in;
	}

    VkResult DescriptorSets() {
        VkResult vulkan_status = VK_SUCCESS;

        array<VkDescriptorSetLayoutBinding, 2> bindings = GlobalLayoutBindings();

        auto global_set_info = CreateDescriptorSetInfo(bindings.data(), bindings.size());

        vulkan_status = vkCreateDescriptorSetLayout(device_, &global_set_info, nullptr, &global_set_layout_);

        auto binding_info = CreateBindingSetInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);

        auto object_set_info = CreateDescriptorSetInfo(&binding_info);

        vulkan_status = vkCreateDescriptorSetLayout(device_, &object_set_info, nullptr, &object_set_layout_);

        return vulkan_status;
    }

    void Clean() {
        vkDestroyDescriptorSetLayout(device_, global_set_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device_, object_set_layout_, nullptr);
    }

    VkDescriptorSetLayout global_set_layout_ = {}, object_set_layout_ = {};

private:
    VkDescriptorSetLayoutCreateInfo CreateDescriptorSetInfo(const VkDescriptorSetLayoutBinding* bindings, const size_t& size = 1) {
        VkDescriptorSetLayoutCreateInfo set_info = {};

        set_info.bindingCount = size;
        set_info.flags = 0;
        set_info.pNext = nullptr;
        set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set_info.pBindings = bindings;

        return set_info;
    }

    array<VkDescriptorSetLayoutBinding, 2> GlobalLayoutBindings() {
        VkDescriptorSetLayoutBinding uniform_layout_binding = {};

        uniform_layout_binding.binding = 0;
        uniform_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uniform_layout_binding.descriptorCount = 1;
        uniform_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        uniform_layout_binding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutBinding sampler_layout_binding = {};

        sampler_layout_binding.binding = 1;
        sampler_layout_binding.descriptorCount = 1;
        sampler_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sampler_layout_binding.pImmutableSamplers = nullptr;
        sampler_layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        return { uniform_layout_binding, sampler_layout_binding };
    }

    VkDescriptorSetLayoutBinding CreateBindingSetInfo(const VkDescriptorType& type, const VkShaderStageFlags& stage_flags, const uint32_t& binding) {
        VkDescriptorSetLayoutBinding object_bind_info = {};

        object_bind_info.binding = binding;
        object_bind_info.descriptorCount = 1;
        object_bind_info.descriptorType = type;
        object_bind_info.pImmutableSamplers = nullptr;
        object_bind_info.stageFlags = stage_flags;

        return object_bind_info;
    }

	VkDevice device_;
};