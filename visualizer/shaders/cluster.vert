#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 modelp;
    mat4 view;
    mat4 proj;
    vec4 lightPos;
    vec4 lightDir;
    vec4 lightColor;
} ubo;

layout(push_constant) uniform PushConstantObject {
    int color;
    mat4 model;
} pco;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 vertNormal;

void main() {
    gl_Position = ubo.proj * ubo.view * pco.model * vec4(inPosition, 1.0);
    vertNormal = inNormal;
    // point size
    gl_PointSize = 10.0;
}