#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPos;

layout(location = 0) out vec3 vertColor;

layout(set = 0, binding = 0) uniform CameraBuffer {
	mat4 view;
	mat4 proj;
	mat4 view_proj;
} CameraData;

layout( push_constant ) uniform constants {
	vec4 data;
	mat4 render_matrix;
} PushConstants;


void main() {
	mat4 transform_matrix = (CameraData.view_proj * PushConstants.render_matrix);

    gl_Position = vec4(inPos, 1.0f);
    vertColor = vec3(1.0f, 1.0f, 1.0f);
}