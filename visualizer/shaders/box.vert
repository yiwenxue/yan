#version 450

layout(location = 0) in vec3 inPosition;

layout(location = 1) in vec4 inMat0;
layout(location = 2) in vec4 inMat1;
layout(location = 3) in vec4 inMat2;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

void main() {
    mat4 inModel = mat4(inMat0, inMat1, inMat2, vec4(0.0, 0.0, 0.0, 1.0));
    inModel = transpose(inModel);

    gl_Position = ubo.proj * ubo.view * inModel * vec4(inPosition, 1.0);
}