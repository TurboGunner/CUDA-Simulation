#pragma once

#include "buffer_helpers.hpp"

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <array>

using std::array;

struct GPUCameraData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 view_proj;
};

struct GPUObjectData {
    glm::mat4 model_matrix;
};

class DescriptorSetHandler {
public:
	DescriptorSetHandler() = default;
	
    DescriptorSetHandler(VkDevice& device_in, VkPhysicalDevice& phys_device_in, VkQueue& queue_in, VkCommandPool& command_pool_in, VkDescriptorPool& descriptor_pool_in, const size_t& max_frames_const_in);

    VkResult DescriptorSets();

    VkResult AllocateDescriptorSets();

    void Clean();

    VkDescriptorSetLayout global_set_layout_ = {}, object_set_layout_ = {};

    vector<GPUObjectData> object_data_;
    vector<VkBuffer> object_buffers_;
    vector<VkDeviceMemory> object_buffer_memory_;
    vector<VkDescriptorSet> object_descriptors_;

    vector<GPUCameraData> camera_data_;
    vector<VkBuffer> camera_buffers_;
    vector<VkDescriptorSet> camera_descriptors_;
    vector<VkDeviceMemory> camera_buffer_memory_;

private:
    VkDescriptorSetLayoutCreateInfo CreateDescriptorSetInfo(const VkDescriptorSetLayoutBinding* bindings, const size_t& size = 1);

    array<VkDescriptorSetLayoutBinding, 2> GlobalLayoutBindings();

    VkDescriptorSetLayoutBinding CreateBindingSetInfo(const VkDescriptorType& type, const VkShaderStageFlags& stage_flags, const uint32_t& binding);

	VkDevice device_;
    VkPhysicalDevice physical_device_;

    VkCommandPool command_pool_;

    VkQueue queue_;

    VkDescriptorPool descriptor_pool_;

    size_t MAX_FRAMES_IN_FLIGHT_;
};