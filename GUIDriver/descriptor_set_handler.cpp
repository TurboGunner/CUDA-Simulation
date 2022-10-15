#include "descriptor_set_handler.hpp"

DescriptorSetHandler::DescriptorSetHandler(VkDevice& device_in, VkPhysicalDevice& phys_device_in, VkQueue& queue_in, VkCommandPool& command_pool_in, VkDescriptorPool& descriptor_pool_in, const size_t& max_frames_const_in) {
    device_ = device_in;
    physical_device_ = phys_device_in;

    command_pool_ = command_pool_in;

    descriptor_pool_ = descriptor_pool_in;

    MAX_FRAMES_IN_FLIGHT_ = max_frames_const_in;

    queue_ = queue_in;
}

VkResult DescriptorSetHandler::DescriptorSets() {
    VkResult vulkan_status = VK_SUCCESS;

    array<VkDescriptorSetLayoutBinding, 2> bindings = GlobalLayoutBindings();
    auto global_set_info = CreateDescriptorSetInfo(bindings.data(), bindings.size());
    vulkan_status = vkCreateDescriptorSetLayout(device_, &global_set_info, nullptr, &global_set_layout_);

    auto binding_info = CreateBindingSetInfo(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
    auto object_set_info = CreateDescriptorSetInfo(&binding_info);

    vulkan_status = vkCreateDescriptorSetLayout(device_, &object_set_info, nullptr, &object_set_layout_);

    return vulkan_status;
}

VkResult DescriptorSetHandler::AllocateDescriptorSets() {
    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;

    VkResult vulkan_status = VK_SUCCESS;

    camera_data_.resize(MAX_FRAMES_IN_FLIGHT_);
    camera_buffers_.resize(MAX_FRAMES_IN_FLIGHT_);
    camera_buffer_memory_.resize(MAX_FRAMES_IN_FLIGHT_);
    camera_descriptors_.resize(MAX_FRAMES_IN_FLIGHT_);

    object_data_.resize(MAX_FRAMES_IN_FLIGHT_);
    object_buffers_.resize(MAX_FRAMES_IN_FLIGHT_);
    object_buffer_memory_.resize(MAX_FRAMES_IN_FLIGHT_);
    object_descriptors_.resize(MAX_FRAMES_IN_FLIGHT_);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT_; i++) {
        BufferHelpers::CreateBufferCross(device_, physical_device_, queue_, command_pool_, &camera_data_[i], camera_buffers_[i], camera_buffer_memory_[i], VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(GPUCameraData));

        const size_t limit = 10000;

        //BufferHelpers::CreateBufferCross(device_, physical_device_, queue_, command_pool_, &object_data_[i], object_buffers_[i], object_buffer_memory_[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(GPUObjectData) * limit);

        VkDescriptorSetAllocateInfo alloc_info = {};

        alloc_info.pNext = nullptr;
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = descriptor_pool_;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &global_set_layout_;

        vulkan_status = vkAllocateDescriptorSets(device_, &alloc_info, &camera_descriptors_[i]);

        VkDescriptorSetAllocateInfo object_alloc_info = {};

        object_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        object_alloc_info.pNext = nullptr;
        object_alloc_info.descriptorPool = descriptor_pool_;
        object_alloc_info.descriptorSetCount = 1;
        object_alloc_info.pSetLayouts = &object_set_layout_;

        //vulkan_status = vkAllocateDescriptorSets(device_, &object_alloc_info, &object_descriptors_[i]);

        VkDescriptorBufferInfo buffer_info = {};

        buffer_info.buffer = camera_buffers_[i];
        buffer_info.offset = 0;
        buffer_info.range = sizeof(GPUCameraData);

        VkDescriptorBufferInfo object_buffer_info = {};

        object_buffer_info.buffer = object_buffers_[i];
        object_buffer_info.offset = 0;
        object_buffer_info.range = sizeof(GPUObjectData) * limit;

        VkWriteDescriptorSet descriptor_write_info = {};

        descriptor_write_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_write_info.pNext = nullptr;

        descriptor_write_info.dstBinding = 0;
        descriptor_write_info.dstSet = camera_descriptors_[i];

        descriptor_write_info.descriptorCount = 1;
        descriptor_write_info.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptor_write_info.pBufferInfo = &buffer_info;

        auto camera_set_write = BufferHelpers::WriteDescriptorSetInfo(camera_descriptors_[i], camera_buffers_[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sizeof(GPUCameraData));
        //auto object_set_write = BufferHelpers::WriteDescriptorSetInfo(object_descriptors_[i], object_buffers_[i], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sizeof(GPUCameraData));

        //array<VkWriteDescriptorSet, 2> descriptor_sets = { camera_set_write, object_set_write };

        vkUpdateDescriptorSets(device_, 1, &descriptor_write_info, 0, nullptr);
    }
    return vulkan_status;
}

void DescriptorSetHandler::Clean() {
    vkDestroyDescriptorSetLayout(device_, global_set_layout_, nullptr);
    vkDestroyDescriptorSetLayout(device_, object_set_layout_, nullptr);
}

VkDescriptorSetLayoutCreateInfo DescriptorSetHandler::CreateDescriptorSetInfo(const VkDescriptorSetLayoutBinding* bindings, const size_t& size) {
    VkDescriptorSetLayoutCreateInfo set_info = {};

    set_info.bindingCount = size;
    set_info.flags = 0;
    set_info.pNext = nullptr;
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_info.pBindings = bindings;

    return set_info;
}

array<VkDescriptorSetLayoutBinding, 2> DescriptorSetHandler::GlobalLayoutBindings() {
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

VkDescriptorSetLayoutBinding DescriptorSetHandler::CreateBindingSetInfo(const VkDescriptorType& type, const VkShaderStageFlags& stage_flags, const uint32_t& binding) {
    VkDescriptorSetLayoutBinding object_bind_info = {};

    object_bind_info.binding = binding;
    object_bind_info.descriptorCount = 1;
    object_bind_info.descriptorType = type;
    object_bind_info.pImmutableSamplers = nullptr;
    object_bind_info.stageFlags = stage_flags;

    return object_bind_info;
}