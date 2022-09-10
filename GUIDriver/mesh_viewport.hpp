#pragma once

#include "mesh_manager.hpp"

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

class MeshViewport {
public:
    MeshViewport() = default;

    MeshPushConstants ViewportRotation(const size_t& frame_index) {
        cam_pos_ = { 0.0f, 0.0f, -2.0f };

        view_ = glm::translate(glm::mat4(1.0f), cam_pos_);

        projection_ = glm::perspective(glm::radians(700.f), 1700.0f / 900.0f, 0.1f, 200.0f);
        projection_[1][1] *= -1;

        glm::mat4 model = glm::rotate(glm::mat4{ 1.0f }, glm::radians(frame_index * 0.1f), glm::vec3(0, 1, 0));

        //Calculates final mesh matrix
        glm::mat4 mesh_matrix = projection_ * view_ * model;

        MeshPushConstants constants;
        constants.render_matrix = mesh_matrix;

        return constants;
    }

private:
    glm::vec3 cam_pos_;
    glm::mat4 view_;
    glm::mat4 projection_;
};