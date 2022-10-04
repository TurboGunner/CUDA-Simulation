#pragma once

#include "mesh_manager.hpp"
#include "descriptor_set_handler.hpp"

#include <vulkan/vulkan.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include <glm/glm.hpp>

#include <glm/gtx/transform.hpp>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <imgui.h>

#include <string>
#include <vector>

using std::vector;
using std::string;

class MeshViewport {
public:
    MeshViewport();

    MeshViewport(VkDevice& device_in);

    void ShiftCamPos(float x = 0.0f, float y = 0.0f, float z = 0.0f, const bool& log = false);

    void ResetCamera();

    void ManipulateCamera();

    MeshPushConstants ViewportRotation(const size_t& frame_index, const size_t& current_frame, DescriptorSetHandler& handler);

private:
    VkDevice device_;

    glm::vec3 cam_pos_;
    glm::mat4 view_;
    glm::mat4 projection_;

    glm::vec3 forward_direction_, right_direction_;
    
    glm::quat quaternion_;

    glm::vec3 default_pos_ = glm::vec3(0.0f, 0.0f, -2.0f);
    glm::vec2 last_mouse_pos;
};