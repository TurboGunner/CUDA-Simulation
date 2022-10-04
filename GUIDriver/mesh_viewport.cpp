#include "mesh_viewport.hpp"

#include "mesh_viewport.hpp"

MeshViewport::MeshViewport() {
    cam_pos_ = default_pos_;

    view_ = glm::mat4{ 1.0f };

    forward_direction_ = glm::vec3(0.0f, 1.0f, 0.0f);
}

MeshViewport::MeshViewport(VkDevice& device_in) {
    device_ = device_in;

    cam_pos_ = default_pos_;

    view_ = glm::mat4{ 1.0f };

    forward_direction_ = glm::vec3(0.0f, 0.0f, 0.0f);
}

void MeshViewport::ShiftCamPos(float x, float y, float z, const bool& log) {
    cam_pos_ += glm::vec3(x, y, z);
    if (log) {
        s_stream << "Shifted camera position by: (" << x << ", " << y << ", " << z << ")!";
        ProgramLog::OutputLine(s_stream);
        s_stream << "Current camera position: (" << cam_pos_.x << ", " << cam_pos_.y << ", " << cam_pos_.z << ").";
        ProgramLog::OutputLine(s_stream);
    }
}

void MeshViewport::ResetCamera() {
    cam_pos_ = default_pos_;
}

void MeshViewport::ManipulateCamera() {
    glm::vec3 up_direction(0.0f, 1.0f, 0.0f);

    float delta_time = ImGui::GetIO().DeltaTime;

    glm::vec2 delta_movement(0.0f, 0.0f);

    bool rotate = ImGui::IsMouseDown(ImGuiMouseButton_Right);

    if (rotate) {
        //ImGui::SetMouseCursor(ImGuiMouseCursor_None);
        SDL_SetRelativeMouseMode(SDL_TRUE);

        auto mouse_pos_temp = ImGui::GetMousePos();
        glm::vec2 mouse_position(mouse_pos_temp.x, mouse_pos_temp.y);

        delta_movement = (mouse_position - last_mouse_pos);

        ProgramLog::OutputLine("Delta Movement: " + std::to_string(forward_direction_.x));

        last_mouse_pos = mouse_position;
    }
    else {
        SDL_SetRelativeMouseMode(SDL_FALSE);
    }

    forward_direction_ = rotate ? forward_direction_ : glm::vec3(0.0f, 1.0f, 0.0f);
    right_direction_ = rotate ? glm::cross(forward_direction_, up_direction) : glm::vec3(1.0f, 0.0f, 0.0f);


    if (ImGui::IsKeyDown(ImGuiKey_W)) {
        cam_pos_ += forward_direction_ * 5.0f * delta_time;
    }
    if (ImGui::IsKeyDown(ImGuiKey_S)) {
        cam_pos_ -= forward_direction_ * 5.0f * delta_time;
    }
    if (ImGui::IsKeyDown(ImGuiKey_A)) {
        cam_pos_ -= right_direction_ * 5.0f * delta_time;
    }
    if (ImGui::IsKeyDown(ImGuiKey_D)) {
        cam_pos_ += right_direction_ * 5.0f * delta_time;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_F, false)) {
        ResetCamera();
    }

    if (delta_movement.x != 0.0f || delta_movement.y != 0.0f) {
        float pitch_delta = delta_movement.y * 0.3f;
        float yaw_delta = delta_movement.x * 0.3f;

        quaternion_ = glm::normalize(glm::cross(glm::angleAxis(-pitch_delta, right_direction_), glm::angleAxis(-yaw_delta, glm::vec3(0.0f, 1.0f, 0.0f))));
        forward_direction_ = glm::rotate(quaternion_, forward_direction_);
    }

}

MeshPushConstants MeshViewport::ViewportRotation(const size_t& frame_index, const size_t& current_frame, DescriptorSetHandler& handler) {
    view_ = glm::lookAt(cam_pos_, forward_direction_, glm::vec3(0.0f, 1.0f, 0.0f));

    projection_ = glm::perspective(glm::radians(70.0f), 1700.0f / 900.0f, 0.1f, 250.0f);

    projection_[1][1] *= -1;

    handler.camera_data_[current_frame].proj = projection_;
    handler.camera_data_[current_frame].view = view_;
    handler.camera_data_[current_frame].view_proj = projection_ * view_;

    void* data = BufferHelpers::MapMemory(device_, &handler.camera_data_[current_frame], sizeof(GPUCameraData), handler.camera_buffer_memory_[current_frame]);

    glm::mat4 model = glm::rotate(glm::mat4{ 1.0f }, glm::radians(frame_index * 0.1f), glm::vec3(0, 1, 0));

    //Calculates the final mesh matrix
    glm::mat4 mesh_matrix = projection_ * view_;

    MeshPushConstants constants;
    constants.render_matrix = mesh_matrix;

    return constants;
}