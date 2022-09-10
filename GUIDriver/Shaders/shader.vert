#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 vertColor;

layout( push_constant ) uniform constants
{
	vec4 data;
	mat4 render_matrix;
} PushConstants;


void main() {
    gl_Position = PushConstants.render_matrix * vec4(inPos, 1.0f);
    vertColor = inColor;
}