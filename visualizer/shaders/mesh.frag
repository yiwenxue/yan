#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 lightPos;
    vec4 lightDir;
    vec4 lightColor;
} ubo;

layout(push_constant) uniform PushConstantObject {
    int color;
} pco;

layout(location = 0) in vec3 vertNormal;

layout(location = 0) out vec4 outColor;

vec4 clusterColor[] = vec4[](
    vec4(1.0, 0.0, 0.0, 1.0), // red
    vec4(0.0, 1.0, 0.0, 1.0), // green
    vec4(0.0, 0.0, 1.0, 1.0), // blue
    vec4(1.0, 1.0, 0.0, 1.0), // yellow
    vec4(1.0, 1.0, 1.0, 1.0)
);

vec4 atmosphereColor = vec4(0.3, 0.3, 0.5, 1.0);

vec4 gammaCorrection(vec4 color) {
    return vec4(pow(color.rgb, vec3(1.0 / 2.2)), color.a);
}

void main() {
    // blinn-phong
    vec4 baseColor = clusterColor[pco.color];

    vec3 lightDir = normalize(ubo.lightDir.xyz);
    vec3 normal = normalize(vertNormal);
    vec3 viewDir = normalize(vec3(
        ubo.view[0][2],
        ubo.view[1][2],
        ubo.view[2][2]
    ));
    vec3 halfDir = normalize(lightDir + viewDir);
    float diffuse = max(dot(normal, lightDir), 0.0);
    float specular = pow(max(dot(normal, halfDir), 0.0), 32.0);
    vec4 specularColor = ubo.lightColor * specular;
    vec4 diffuseColor = ubo.lightColor * diffuse;
    vec4 ambientColor = vec4(0.1, 0.1, 0.1, 1.0);
    vec4 color = (ambientColor + diffuseColor + specularColor) * baseColor;

    // gamma correction
    outColor = gammaCorrection(color);
}
