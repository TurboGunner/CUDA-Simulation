#pragma once

#include "buffer_helpers.hpp"

#include <vulkan/vulkan.h>

struct GPUCameraData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 view_proj;
};

class DescriptorSetHandler {
public:
	DescriptorSetHandler() = default;
	
	DescriptorSetHandler(VkDevice& device_in, VkPhysicalDevice& phys_device_in, VkQueue& queue_in, VkCommandPool& command_pool_in, VkDescriptorPool& descriptor_pool_in, const size_t& max_frames_const_in) {
		device_ = device_in;
        physical_device_ = phys_device_in;

        command_pool_ = command_pool_in;

        descriptor_pool_ = descriptor_pool_in;

        MAX_FRAMES_IN_FLIGHT_ = max_frames_const_in;

        queue_ = queue_in;
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

    VkResult AllocateDescriptorSets() {
        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_memory;

        VkResult vulkan_status = VK_SUCCESS;

        camera_data_.resize(MAX_FRAMES_IN_FLIGHT_);
        camera_buffers_.resize(MAX_FRAMES_IN_FLIGHT_);
        camera_buffer_memory_.resize(MAX_FRAMES_IN_FLIGHT_);
        global_descriptors_.resize(MAX_FRAMES_IN_FLIGHT_);

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT_; i++) {
            vulkan_status = BufferHelpers::CreateBuffer(device_, physical_device_, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, sizeof(GPUCameraData), staging_buffer, staging_buffer_memory);

            void* data = BufferHelpers::MapMemory(device_, &camera_data_[i], sizeof(GPUCameraData), staging_buffer_memory);

            vulkan_status = BufferHelpers::CreateBuffer(device_, physical_device_, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sizeof(GPUCameraData), camera_buffers_[i], camera_buffer_memory_[i]);

            vulkan_status = BufferHelpers::CopyBuffer(device_, queue_, command_pool_, staging_buffer, camera_buffers_[i], sizeof(GPUCameraData));

            VkDescriptorSetAllocateInfo alloc_info = {};

            alloc_info.pNext = nullptr;
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = descriptor_pool_;
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &global_set_layout_;

            vulkan_status = vkAllocateDescriptorSets(device_, &alloc_info, &global_descriptors_[i]);

            VkDescriptorBufferInfo buffer_info = {};

            buffer_info.buffer = camera_buffers_[i];
            buffer_info.offset = 0;
            buffer_info.range = sizeof(GPUCameraData);

            VkWriteDescriptorSet descriptor_write_info = {};

            descriptor_write_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_write_info.pNext = nullptr;

            descriptor_write_info.dstBinding = 0;
            descriptor_write_info.dstSet = global_descriptors_[i];

            descriptor_write_info.descriptorCount = 1;
            descriptor_write_info.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptor_write_info.pBufferInfo = &buffer_info;

            //auto descriptor_set_write = BufferHelpers::WriteDescriptorSetInfo(global_descriptors_[i], camera_buffers_[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sizeof(GPUCameraData));

            vkUpdateDescriptorSets(device_, 1, &descriptor_write_info, 0, nullptr);

            vkDestroyBuffer(device_, staging_buffer, nullptr);
            vkFreeMemory(device_, staging_buffer_memory, nullptr);
        }
        return vulkan_status;
    }

    void Clean() {
        vkDestroyDescriptorSetLayout(device_, global_set_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device_, object_set_layout_, nullptr);
    }

    VkDescriptorSetLayout global_set_layout_ = {}, object_set_layout_ = {};

    vector<GPUCameraData> camera_data_;
    vector<VkBuffer> camera_buffers_;
    vector<VkDescriptorSet> global_descriptors_;
    vector<VkDeviceMemory> camera_buffer_memory_;

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
    VkPhysicalDevice physical_device_;

    VkCommandPool command_pool_;

    VkQueue queue_;

    VkDescriptorPool descriptor_pool_;

    size_t MAX_FRAMES_IN_FLIGHT_;
};