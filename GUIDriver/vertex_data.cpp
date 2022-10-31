#include "vertex_data.hpp"
#include "../Meshing/vector_cross.cuh"

__host__ __device__ Vertex::Vertex(float pos_x, float pos_y, float pos_z, float r, float g, float b, float n_x, float n_y, float n_z) {
    pos = glm::vec3(pos_x, pos_y, pos_z);
    color = glm::vec3(r, g, b);
    glm::vec3(n_x, n_y, n_z);
}

void Vertex::RGBAdjust(float r, float g, float b, float size) {
    if (size <= 0.0f) {
        ProgramLog::OutputLine("Invalid input size of " + std::to_string(size) + "! Must be greater than 0.");
        return;
    }

    if ((r > size || g > size || b > size) && (r < 0 || g < 0 || b < 0)) {
        s_stream << "Invalid input size of color (" << r << ", " << g << ", " << b << ")! Must be greater than 0 and less than size.";
        ProgramLog::OutputLine(s_stream);
        return;
    }

    size = int(size);

    r /= size;
    g /= size;
    b /= size;

    color = glm::vec3(r, g, b);
}

vector<VkVertexInputBindingDescription> Vertex::GetBindingDescription() {
    VkVertexInputBindingDescription binding_description = {};

    binding_description.binding = 0;
    binding_description.stride = sizeof(Vector3D);
    binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    return { binding_description };
}

vector<VkVertexInputAttributeDescription> Vertex::GetAttributeDescriptions() {
    vector<VkVertexInputAttributeDescription> attribute_descriptions;

    VkVertexInputAttributeDescription pos_description = {};
    pos_description.binding = 0;
    pos_description.location = 0;
    pos_description.format = VK_FORMAT_R32G32B32_SFLOAT; //NOTE
   //pos_description.offset = offsetof(Vector3D, Vector3D::dim);
    pos_description.offset = 0;

    VkVertexInputAttributeDescription normal_description = {};
    normal_description.binding = 0;
    normal_description.location = 1;
    normal_description.format = VK_FORMAT_R32G32B32_SFLOAT;
    normal_description.offset = offsetof(Vertex, normal);

    VkVertexInputAttributeDescription color_description = {};
    color_description.binding = 0;
    color_description.location = 2;
    color_description.format = VK_FORMAT_R32G32B32_SFLOAT;
    color_description.offset = offsetof(Vertex, color);

    //attribute_descriptions.insert(attribute_descriptions.end(), { pos_description, normal_description, color_description });
    attribute_descriptions.insert(attribute_descriptions.end(), { pos_description });

    return attribute_descriptions;
}

Vertex& Vertex::operator=(const Vertex& copy) {
    pos = copy.pos;
    color = copy.color;

    return *this;
}

bool Vertex::operator==(Vertex const& compare) {
    bool pos_check = pos.x == compare.pos.x && pos.y == compare.pos.y && pos.z == compare.pos.z;
    bool color_check = color.x == compare.color.x && color.y == compare.color.y && color.z == compare.color.z;
    bool normal_check = normal.x == compare.normal.x && normal.y == compare.normal.y && normal.z == compare.normal.z;

    return pos_check && color_check && normal_check;
}