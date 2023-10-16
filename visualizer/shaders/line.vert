#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 vertNormal;

void main() {
    vec3 pos = inPosition + inNormal * 0.0001;
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(pos, 1.0);
    vertNormal = inNormal;
}