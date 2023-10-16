#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 lightPos;
    vec4 lightDir;
    vec4 lightColor;
} ubo;

layout(location = 0) in vec3 vertNormal;

layout(location = 0) out vec4 outColor;

// alert color
vec3 baseColor = vec3(1.0, 1.0, 1.0);

void main() {
    // wireframe with strange colot
    outColor = vec4(baseColor, 1.0);
}