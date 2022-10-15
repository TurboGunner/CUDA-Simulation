#include "mesh_manager.hpp"

VertexData::VertexData(VkDevice& device_in, VkPhysicalDevice& phys_device_in, VkQueue& queue_in, const size_t& max_frames_const_in, const bool& binding_mode_in) {
    device_ = device_in;
    physical_device_ = phys_device_in;
    queue_ = queue_in;

    /*
    vector<Vertex> vertices_in = {
        Vertex(0.5f, -0.5f, 0.0f, 1.0f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f),
        Vertex(0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 1.0f, 0.5f, 0.5f, 0.5f),
        Vertex(-1.5f, 1.0f, 1.0f, 1.0f, 0.5f, 1.0f, 0.5f, 0.5f, 0.5f),
        Vertex(-1.5f, -1.5f, -1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f),
        Vertex(1.0f, -1.5f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f),
        Vertex(1.0f, 1.0f, 1.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f)
    };

    vertices.Push(vertices_in);
    size_ = sizeof(vertices[0]) * vertices.Size();
    */

    index_binding_mode_ = binding_mode_in;

    MAX_FRAMES_IN_FLIGHT_ = max_frames_const_in;
}

void VertexData::BindPipeline(VkCommandBuffer& command_buffer, VkCommandPool& command_pool) {
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer_, offsets);
    if (index_binding_mode_) {
        vkCmdBindIndexBuffer(command_buffer, index_buffer_, 0, VK_INDEX_TYPE_UINT16);
    }
}

VkResult VertexData::Initialize(VkCommandPool& command_pool) {
    VkResult vulkan_status = VK_SUCCESS;

    if (vertices.SyncMode()) {
        vertex_buffer_ = vertices.sync_data_.buffer;
        vertex_buffer_memory_ = vertices.sync_data_.buffer_memory;

        return vulkan_status;
    }

    void* data = nullptr;

    vulkan_status = BufferHelpers::CreateBufferCross(device_, physical_device_, queue_, command_pool, data, index_buffer_, index_buffer_memory_, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, size_);

    return vulkan_status;
}

VkResult VertexData::InitializeIndex(VkCommandPool& command_pool) {
    VkResult vulkan_status = VK_SUCCESS;

    if (vertices.SyncMode() || !index_binding_mode_) {
        return vulkan_status;
    }

    const vector<uint16_t> indices = { 0, 1, 2, 0, 3, 2 }; //Make this a data member for the vert?

    void* data = nullptr;

    vulkan_status = BufferHelpers::CreateBufferCross(device_, physical_device_, queue_, command_pool, data, index_buffer_, index_buffer_memory_, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, size_);

    return vulkan_status;
}

bool VertexData::IndexBindingMode() const {
    return index_binding_mode_;
}

void VertexData::Clean() {
    vkDestroyBuffer(device_, vertex_buffer_, nullptr);
    vkDestroyBuffer(device_, index_buffer_, nullptr);

    vkFreeMemory(device_, vertex_buffer_memory_, nullptr);
    vkFreeMemory(device_, index_buffer_memory_, nullptr);
}