#pragma once

#include "mesh_manager.hpp"
#include "descriptor_set_handler.hpp"

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <vector>

using std::vector;

class MeshViewport {
public:
    MeshViewport() = default;

    MeshViewport(VkDevice& device_in, VkDescriptorPool& descriptor_pool_in) {
        device_ = device_in;
        descriptor_pool_ = descriptor_pool_in;
    }

    void ShiftCamPos(float x = 0.0f, float y = 0.0f, float z = 0.0f) {
        cam_pos_ += glm::vec3(x, y, z);
    }

    MeshPushConstants ViewportRotation(const size_t& frame_index, const size_t& current_frame, DescriptorSetHandler& handler) {
        cam_pos_ = { 0.0f, 0.0f, -2.0f };

        view_ = glm::translate(glm::mat4(1.0f), cam_pos_);

        projection_ = glm::perspective(glm::radians(70.0f), 1700.0f / 900.0f, 0.1f, 250.0f);
        projection_[1][1] *= -1;

        handler.camera_data_[current_frame].proj = projection_;
        handler.camera_data_[current_frame].view = view_;
        handler.camera_data_[current_frame].view_proj = projection_ * view_;

        void* data = BufferHelpers::MapMemory(device_, &handler.camera_data_[current_frame], sizeof(GPUCameraData), handler.camera_buffer_memory_[current_frame]);

        glm::mat4 model = glm::rotate(glm::mat4{ 1.0f }, glm::radians(frame_index * 0.1f), glm::vec3(0, 1, 0));

        //Calculates the final mesh matrix
        glm::mat4 mesh_matrix = projection_ * view_ * model;

        MeshPushConstants constants;
        constants.render_matrix = mesh_matrix;

        return constants;
    }

private:
    VkDevice device_;
    VkDescriptorPool descriptor_pool_;

    glm::vec3 cam_pos_;
    glm::mat4 view_;
    glm::mat4 projection_;
};