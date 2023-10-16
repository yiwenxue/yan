#include "render.h"
#include "utility.hpp"
#include "viewer.h"
#include <cmath>
#include <glm/ext/matrix_transform.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <limits>
#include <numbers>
#include <numeric>
#include <ranges>
#include <vulkan/vulkan_core.h>

#include <glm/gtx/matrix_decompose.hpp>

void bbox_gui(Bounds &bounds) noexcept {
    ImGui::Text("Min: (%.2f, %.2f, %.2f)", bounds.min[0], bounds.min[1], bounds.min[2]);
    ImGui::Text("Max: (%.2f, %.2f, %.2f)", bounds.max[0], bounds.max[1], bounds.max[2]);
}

float toRadian(float degree) {
    return degree / 180.0f * std::numbers::pi_v<float>;
}

void mesh_gui(Scene &scene, uint32_t i) {
    auto &mesh = scene.meshes[i];

    bool enabled = scene.flags[i] & Scene::Mask::enabled;
    if (ImGui::Checkbox("Enable", &enabled)) {
        if (enabled) {
            scene.flags[i] |= Scene::Mask::enabled;
        } else {
            scene.flags[i] &= ~Scene::Mask::enabled;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Only")) {
        for (auto &m : scene.flags) {
            m &= ~Scene::Mask::enabled;
        }
        scene.flags[i] |= Scene::Mask::enabled;
    }
    ImGui::SameLine();
    if (ImGui::Button("ALL")) {
        for (auto &m : scene.flags) {
            m |= Scene::Mask::enabled;
        }
    }
    // present the model matrix
    glm::vec3 pos, scale, skew;
    glm::vec4 perspective;
    glm::quat rotation;
    glm::decompose(mesh.pso.model, scale, rotation, pos, skew, perspective);
    glm::vec3 euler = glm::eulerAngles(rotation);

    ImGui::Text("Model Transform");
    ImGui::DragFloat3("Position", glm::value_ptr(pos), 0.1f);
    ImGui::DragFloat3("Scale", glm::value_ptr(scale), 0.1f);
    ImGui::DragFloat3("Rotation", glm::value_ptr(euler), 0.1f);
    rotation       = glm::quat(euler);
    mesh.pso.model = glm::translate(glm::mat4(1.0f), pos) * glm::mat4_cast(rotation)
                     * glm::scale(glm::mat4(1.0f), scale);
    ImGui::Separator();
    // mesh detail
    ImGui::Text("Triangle Count: %d", mesh.index_count / 3);
    ImGui::Text("Index Count: %d", mesh.index_count);
    ImGui::Text("Vertex Count: %d", mesh.vertex_count);
    // raster type
    ImGui::Combo("Raster", (int *) &mesh.raster, "Filled\0Wireframe\0Point\0\0");
    // color is enum
    ImGui::Combo("Color", (int *) &mesh.pso.clusterColorId,
                 "Red\0Green\0Blue\0Yellow\0Magenta\0Cyan\0White\0\0");

    if (mesh.raster == RasterizeType::WIREFRAME) {
        ImGui::DragFloat("Line Width", &mesh.lineWidth, 0.1f, 0.0f, 5.0f);
    }
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Data")) {
        bbox_gui(mesh.bound);
        // plot a table to show position data, showing 3 columns
        ImGui::Columns(4, "mycolumns"); // 3-ways, no border
        ImGui::Text("Position");
        ImGui::Separator();
        ImGui::Text("Id");
        ImGui::NextColumn();
        ImGui::Text("X");
        ImGui::NextColumn();
        ImGui::Text("Y");
        ImGui::NextColumn();
        ImGui::Text("Z");
        ImGui::NextColumn();
        ImGui::Separator();

        auto clusterData = scene.clusters[i];
        for (auto i : std::views::iota(0U, clusterData.vertices.size())) {
            const auto &v = clusterData.vertices[i];
            ImGui::Text("%d", i);
            ImGui::NextColumn();
            ImGui::Text("%.2f", v.data[0]);
            ImGui::NextColumn();
            ImGui::Text("%.2f", v.data[1]);
            ImGui::NextColumn();
            ImGui::Text("%.2f", v.data[2]);
            ImGui::NextColumn();
        }
        ImGui::Columns(1);
    }
}

void scene_gui(Scene &scene) noexcept {
    // check shortcut key press
    static uint32_t nearest = -1;
    if (ImGui::IsKeyPressed(ImGuiKey_C) && !isGuiHovered()) {
        // perform ray cast
        // cursor pos in NDC
        float x = lastX, y = lastY;
        x = x / WINDOW_WIDTH * 2.0 - 1.0;
        y = y / WINDOW_HEIGHT * 2.0 - 1.0;
        y = -y;

        float     frameY = std::tan(toRadian(c_fov / 2.0f));
        float     frameX = frameY * WINDOW_WIDTH / WINDOW_HEIGHT;
        glm::vec3 r_dir  = glm::normalize(c_front) + glm::normalize(c_right) * x * frameX
                          + glm::normalize(c_up_p) * y * frameY;
        r_dir = glm::normalize(r_dir);

        Vec3 ori{c_pos.x, c_pos.y, c_pos.z};
        Vec3 dir = {r_dir.x, r_dir.y, r_dir.z};

        float t = 0;

        nearest = ray_cast_nearest(scene, ori, dir, t);

        if (nearest != -1) {
            auto point = std::find_if(scene.meshes.begin(), scene.meshes.end(),
                                      [&](const auto &mesh) { return mesh.name == "Sphere"; });

            if (point != scene.meshes.end()) {
                glm::vec3 pos, scale, skew;
                glm::vec4 perspective;
                glm::quat rotation;
                glm::decompose(point->pso.model, scale, rotation, pos, skew, perspective);
                glm::vec3 euler = glm::eulerAngles(rotation);

                pos              = c_pos + r_dir * t;
                rotation         = glm::quat(euler);
                point->pso.model = glm::translate(glm::mat4(1.0f), pos) * glm::mat4_cast(rotation)
                                   * glm::scale(glm::mat4(1.0f), scale);

                point->pso.clusterColorId = static_cast<uint32_t>(ClusterColor::YELLOW);
            }
        }
    }

    if (nearest != -1) {
        if (ImGui::CollapsingHeader("Nearest", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SameLine();
            // name at the end, calculate the pos
            auto &mesh     = scene.meshes[nearest];
            auto  texWidth = ImGui::CalcTextSize(mesh.name.c_str()).x;
            auto  avail    = ImGui::GetContentRegionAvail().x;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + avail - texWidth);
            ImGui::Text("%s", mesh.name.c_str());
            mesh_gui(scene, nearest);
        }
    }

    if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("Scene");
        // scene detail
        ImGui::Text("Cluster Count: %lld", scene.meshes.size());
        // scene color
        for (uint32_t i = 0; i < scene.meshes.size(); i++) {
            auto &mesh = scene.meshes[i];

            mesh.ghostPso    = mesh.pso;
            mesh.ghostRaster = mesh.raster;
            scene.flags[i] &= ~Scene::Mask::bounded;

            ImGui::PushID(i);
            if (ImGui::TreeNode(mesh.name.c_str())) {
                mesh_gui(scene, i);
                ImGui::TreePop();
            }
            ImGui::PopID();

            if (ImGui::IsItemHovered()) {
                mesh.ghostRaster = RasterizeType::WIREFRAME;
                scene.flags[i] |= Scene::Mask::bounded;
            }
        }
        ImGui::PopID();
    }
}

bool ray_aabb(const Vec3 &ori, const Vec3 &dir, const Bounds &bound, float tmin, float tmax) {
    float tmin_ = tmin, tmax_ = tmax;
    for (int i = 0; i < 3; i++) {
        float invD = 1.0f / dir.data[i];
        float t0   = (bound.min[i] - ori.data[i]) * invD;
        float t1   = (bound.max[i] - ori.data[i]) * invD;

        tmin_ = std::max(tmin_, std::min(t0, t1));
        tmax_ = std::min(tmax_, std::max(t0, t1));
    }

    return tmin_ <= tmax_ && tmin_ >= tmin && tmax_ <= tmax;
}

bool ray_triangle(const Vec3 &ori, const Vec3 &dir, const Vec3 &v0, const Vec3 &v1, const Vec3 &v2,
                  float &t, float &u, float &v) {
    Vec3           e1      = v1 - v0;
    Vec3           e2      = v2 - v0;
    Vec3           p       = cross(dir, e2);
    float          a       = dot(e1, p);
    constexpr auto EPSILON = std::numeric_limits<float>::epsilon();
    if (a > -EPSILON && a < EPSILON) {
        return false;
    }
    float f = 1.0f / a;
    Vec3  s = ori - v0;
    u       = f * dot(s, p);
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    Vec3 q = cross(s, e1);
    v      = f * dot(dir, q);
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    t = f * dot(e2, q);
    return t > EPSILON;
}

uint32_t ray_cast_nearest(Scene &scene, Vec3 ori, Vec3 dir, float &t, float tmin, float tmax) {
    uint32_t nearest = -1;
    float    t0      = INFINITY;
    for (auto i : std::views::iota(0u, scene.meshes.size())) {
        auto &mesh = scene.meshes[i];
        if (mesh.topology != TopologyType::TRIANGLE || !(scene.flags[i] & Scene::Mask::enabled)) {
            continue;
        }
        auto &cluster = scene.clusters[i];
        if (ray_aabb(ori, dir, mesh.bound, tmin, tmax)) {
            for (uint32_t j = 0; j < mesh.index_count; j += 3) {
                float t, u, v;
                if (ray_triangle(ori, dir, cluster.vertices[cluster.indices[j]],
                                 cluster.vertices[cluster.indices[j + 1]],
                                 cluster.vertices[cluster.indices[j + 2]], t, u, v)) {
                    if (t < tmax && t > tmin && t < t0) {
                        t0      = t;
                        nearest = i;
                    }
                }
            }
        }
    }
    t = t0;
    return nearest;
}

void mesh_gui(MeshRender &instance) noexcept {
    if (ImGui::CollapsingHeader(instance.name.c_str())) {
        // mesh detail
        ImGui::Text("Triangle Count: %d", instance.index_count / 3);
        ImGui::Text("Index Count: %d", instance.index_count);
        ImGui::Text("Vertex Count: %d", instance.vertex_count);
        // raster type
        ImGui::Combo("Raster", (int *) &instance.raster, "Filled\0Wireframe\0\0");
        // color is enum
        ImGui::Combo("Color", (int *) &instance.pso.clusterColorId,
                     "Red\0Green\0Blue\0Yellow\0Magenta\0Cyan\0White\0\0");
    }
}

Render::Render(std::string_view title) {
    window = createWindow(title.data(), WINDOW_WIDTH, WINDOW_HEIGHT);
    if (!window) {
        return;
    }
    glfwSetWindowAttrib(window, GLFW_RESIZABLE, GLFW_FALSE);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetScrollCallback(window, scrollCallback);

    std::vector<const char *> instanceExtensions = getGlfwExtensions();

    instance = createInstance(instanceExtensions);
    surface  = createSurface(instance, window);

    std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    std::vector<uint32_t> queueFamilyIndices;
    physicalDevice = pickPhysicalDevice(instance, surface, deviceExtensions);
    queueFamilyIndices.push_back(selectQueueFamily(surface, physicalDevice));

    device    = createDevice(instance, physicalDevice, queueFamilyIndices, deviceExtensions);
    allocator = createAllocator(instance, physicalDevice, device);

    auto swapchainSup            = getSwapChainSupport(physicalDevice, surface);
    surfaceFormat                = chooseSwapSurfaceFormat(swapchainSup.formats);
    swapchainExtent              = chooseSwapExtent(swapchainSup.capabilities, window);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapchainSup.presentModes);
    swapChain = createSwapchain(physicalDevice, device, surface, queueFamilyIndices[0],
                                swapchainExtent, surfaceFormat, presentMode);

    graphicsQueue = getQueue(device, queueFamilyIndices[0]);
    commandPool   = createCommandPool(device, queueFamilyIndices[0]);
    for (size_t i = 0; i < MAX_IN_FLIGHT_FRAMES; ++i) {
        commandBuffers[i] = createCommandBuffer(device, commandPool);
    }

    depthFormat = pickDepthFormat(physicalDevice);

    renderPass = createRenderPass(device, surfaceFormat.format, depthFormat);

    vkGetSwapchainImagesKHR(device, swapChain, &swapchainImageCount, nullptr);
    swapchainImages.resize(swapchainImageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &swapchainImageCount, swapchainImages.data());

    swapchainImageViews.resize(swapchainImageCount);
    depthImages.resize(swapchainImageCount);
    depthImageMemories.resize(swapchainImageCount);
    depthImageViews.resize(swapchainImageCount);
    for (size_t i = 0; i < swapchainImageCount; ++i) {
        swapchainImageViews[i] = createImageView(device, swapchainImages[i], surfaceFormat.format,
                                                 VK_IMAGE_ASPECT_COLOR_BIT);
        depthImages[i]         = createImage(physicalDevice, device, swapchainExtent.width,
                                     swapchainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL,
                                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImageMemories[i]);
        depthImageViews[i]
            = createImageView(device, depthImages[i], depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    framebuffers.resize(swapchainImageCount);
    for (size_t i = 0; i < swapchainImageCount; ++i) {
        framebuffers[i] = createFramebuffer(device, renderPass, swapchainImageViews[i],
                                            depthImageViews[i], swapchainExtent);
    }

    for (size_t i = 0; i < MAX_IN_FLIGHT_FRAMES; ++i) {
        inFlightFences[i]           = createFence(device);
        imageAvailableSemaphores[i] = createSemaphore(device);
        renderFinishedSemaphores[i] = createSemaphore(device);
        uniformBuffers[i]           = createBuffer(allocator, sizeof(UniformBufferObject),
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VMA_MEMORY_USAGE_CPU_TO_GPU, uniformBufferMemorys[i]);
        vmaMapMemory(allocator, uniformBufferMemorys[i], &uniformBufferMapped[i]);
    }

    imGuiDescriptorPool = createDescriptorPool(device, 50, 100);
    createImGuiContext(window, instance, physicalDevice, device, graphicsQueue,
                       queueFamilyIndices[0], commandPool, imGuiDescriptorPool, surface, renderPass,
                       swapchainImageCount);

    std::vector<VkDescriptorPoolSize> poolSize{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = poolSize.size();
    poolInfo.pPoolSizes    = poolSize.data();
    poolInfo.maxSets       = MAX_IN_FLIGHT_FRAMES * (maxUboCount + maxImageCount);
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }

    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding         = 0;
    uboLayoutBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo uboLayoutInfo{};
    uboLayoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    uboLayoutInfo.bindingCount = 1;
    uboLayoutInfo.pBindings    = &uboLayoutBinding;
    if (vkCreateDescriptorSetLayout(device, &uboLayoutInfo, nullptr, &descriptorSetLayout)
        != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    std::vector<VkDescriptorSetLayout> layouts(MAX_IN_FLIGHT_FRAMES, descriptorSetLayout);
    VkDescriptorSetAllocateInfo        allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = descriptorPool;
    allocInfo.descriptorSetCount = layouts.size();
    allocInfo.pSetLayouts        = layouts.data();
    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < MAX_IN_FLIGHT_FRAMES; i++) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = Render::uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range  = sizeof(Render::UniformBufferObject);

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet           = descriptorSets[i];
        descriptorWrite.dstBinding       = 0;
        descriptorWrite.dstArrayElement  = 0;
        descriptorWrite.descriptorType   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.descriptorCount  = 1;
        descriptorWrite.pBufferInfo      = &bufferInfo;
        descriptorWrite.pImageInfo       = nullptr; // Optional
        descriptorWrite.pTexelBufferView = nullptr; // Optional

        vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset     = 0;
    pushConstantRange.size       = sizeof(PushConstantObject);

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount         = 1;
    pipelineLayoutCreateInfo.pSetLayouts            = &descriptorSetLayout;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstantRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout)
        != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }
}

Render::Pipeline Render::acquirePipeline(RasterizeType raster, TopologyType topology,
                                         std::string vertShader, std::string fragShader) {
    PipelineKey key{raster, topology, vertShader, fragShader};
    if (auto it = pipelines.find(key); it != pipelines.end()) {
        return it->second;
    }

    // create pipeline
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    auto bindingDescription    = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount
        = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions   = &bindingDescription;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = topology == TopologyType::LINE ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST
                             : topology == TopologyType::TRIANGLE
                                 ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
                                 : VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = (float) swapchainExtent.width;
    viewport.height   = (float) swapchainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapchainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports    = &viewport;
    viewportState.scissorCount  = 1;
    viewportState.pScissors     = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable        = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode             = raster == RasterizeType::WIREFRAME ? VK_POLYGON_MODE_LINE
                                         : raster == RasterizeType::FILLED  ? VK_POLYGON_MODE_FILL
                                                                            : VK_POLYGON_MODE_POINT;
    rasterizer.lineWidth               = 1.0f;
    rasterizer.cullMode                = VK_CULL_MODE_NONE;
    rasterizer.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable         = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional
    rasterizer.depthBiasClamp          = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor    = 0.0f; // Optional

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable   = VK_FALSE;
    multisampling.rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading      = 1.0f;     // Optional
    multisampling.pSampleMask           = nullptr;  // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable      = VK_FALSE; // Optional

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                          | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable         = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;      // Optional
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;      // Optional

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType             = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable     = VK_FALSE;
    colorBlending.logicOp           = VK_LOGIC_OP_COPY; // Optional
    colorBlending.attachmentCount   = 1;
    colorBlending.pAttachments      = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable       = VK_TRUE;
    depthStencil.depthWriteEnable      = VK_TRUE;
    depthStencil.depthCompareOp        = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable     = VK_FALSE;

    std::vector<VkDynamicState> dynamicStates
        = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_LINE_WIDTH};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = dynamicStates.size();
    dynamicState.pDynamicStates    = dynamicStates.data();

    const auto vertCode = readFile(vertShader);
    const auto fragCode = readFile(fragShader);

    VkShaderModule vertShaderModule = createShaderModule(device, vertCode);
    VkShaderModule fragShaderModule = createShaderModule(device, fragCode);

    VkPipelineShaderStageCreateInfo shaderStages[2] = {};
    shaderStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].module = vertShaderModule;
    shaderStages[0].pName  = "main";
    shaderStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module = fragShaderModule;
    shaderStages[1].pName  = "main";

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount          = 2;
    pipelineInfo.pStages             = shaderStages;
    pipelineInfo.pVertexInputState   = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisampling;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlending;
    pipelineInfo.pDynamicState       = &dynamicState;
    pipelineInfo.layout              = pipelineLayout;
    pipelineInfo.renderPass          = renderPass;
    pipelineInfo.subpass             = 0;

    VkPipeline pipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline)
        != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);

    pipelines[key] = Pipeline{vertShader, fragShader, pipeline};

    return pipelines[key];
}

Render::~Render() {
    vkDestroyDescriptorPool(device, imGuiDescriptorPool, nullptr);

    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

    for (auto &[_, pipeline] : pipelines) {
        vkDestroyPipeline(device, pipeline.pipeline, nullptr);
    }

    cleanupImGuiContext();

    vkDestroyRenderPass(device, renderPass, nullptr);

    for (size_t i = 0; i < MAX_IN_FLIGHT_FRAMES; ++i) {
        vmaUnmapMemory(allocator, uniformBufferMemorys[i]);
        vmaDestroyBuffer(allocator, uniformBuffers[i], uniformBufferMemorys[i]);
        vkDestroyFence(device, inFlightFences[i], nullptr);
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
    }

    for (size_t i = 0; i < swapchainImageCount; ++i) {
        vkDestroyFramebuffer(device, framebuffers[i], nullptr);
    }

    for (size_t i = 0; i < swapchainImageCount; ++i) {
        vkDestroyImageView(device, depthImageViews[i], nullptr);
        vkDestroyImage(device, depthImages[i], nullptr);
        vkFreeMemory(device, depthImageMemories[i], nullptr);
    }

    for (size_t i = 0; i < swapchainImageCount; ++i) {
        vkDestroyImageView(device, swapchainImageViews[i], nullptr);
    }

    vmaDestroyAllocator(allocator);

    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroySwapchainKHR(device, swapChain, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);

    cleanupWindow(window);
}

void Render::updateUbo() {
    ubo.model      = glm::mat4(1.0);
    ubo.view       = g_view;
    ubo.proj       = g_proj;
    ubo.lightPos   = g_light_pos;
    ubo.lightDir   = g_light_dir;
    ubo.lightColor = g_light_col;

    memcpy(uniformBufferMapped[currentBuffer], &ubo, sizeof(UniformBufferObject));
}

int Render::loop(RenderFunc preProcess, RenderFunc render, RenderFunc postProcess) {
    while (!glfwWindowShouldClose(window)) {
        vkWaitForFences(device, 1, &inFlightFences[currentBuffer], VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &inFlightFences[currentBuffer]);

        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
                                                imageAvailableSemaphores[currentBuffer],
                                                VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            std::cout << "swapchain out of date" << std::endl;
            break;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            std::cout << "failed to acquire swapchain image" << std::endl;
            return -1;
        }

        static auto lastTime = std::chrono::steady_clock::now();
        auto        now      = std::chrono::steady_clock::now();

        dt = std::chrono::duration<float, std::chrono::seconds::period>(now - lastTime).count();

        lastTime = now;
        fps      = lerp(fps, 1.0f / dt, 0.1f);
        ms       = lerp(ms, dt * 1000.0f, 0.1f);

        auto framebuffer   = framebuffers[imageIndex];
        auto commandBuffer = commandBuffers[currentBuffer];

        updateUbo();
        preProcess();

        VkCommandBufferBeginInfo commandBufferBeginInfo{};
        commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkResetCommandBuffer(commandBuffer, 0);
        vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);

        VkClearValue clearValues[2] = {};
        clearValues[0].color        = {0.4f, 0.4f, 0.6f, 1.0f};
        clearValues[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo renderPassBeginInfo{};
        renderPassBeginInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassBeginInfo.renderPass        = renderPass;
        renderPassBeginInfo.framebuffer       = framebuffer;
        renderPassBeginInfo.renderArea.offset = {0, 0};
        renderPassBeginInfo.renderArea.extent = swapchainExtent;
        renderPassBeginInfo.clearValueCount   = 2;
        renderPassBeginInfo.pClearValues      = clearValues;

        vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport{};
        viewport.x        = 0.0f;
        viewport.y        = 0.0f;
        viewport.width    = static_cast<float>(swapchainExtent.width);
        viewport.height   = static_cast<float>(swapchainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapchainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // draw
        render();

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

        VkSubmitInfo submitInfo{};
        submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &commandBuffer;

        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores    = &imageAvailableSemaphores[currentBuffer];
        submitInfo.pWaitDstStageMask  = waitStages;

        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores    = &renderFinishedSemaphores[currentBuffer];

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentBuffer]);

        // present image
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores    = &renderFinishedSemaphores[currentBuffer];

        VkSwapchainKHR swapChains[] = {swapChain};

        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains    = swapChains;
        presentInfo.pImageIndices  = &imageIndex;

        result = vkQueuePresentKHR(graphicsQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            std::cout << "swapchain out of date" << std::endl;
            break;
        } else if (result != VK_SUCCESS) {
            std::cout << "failed to present swapchain image" << std::endl;
            return -1;
        }

        currentBuffer = (currentBuffer + 1) % MAX_IN_FLIGHT_FRAMES;
        frameCount++;

        postProcess();

        glfwPollEvents();
    }
    return vkDeviceWaitIdle(device);
}

MeshRender::MeshRender(std::string name, Render &render, std::span<Vertex> vertices,
                       std::span<uint32_t> indices, TopologyType topology, RasterizeType raster) :
    name(name), renderer(render), topology(topology), raster(raster) {
    auto allocator = render.allocator;

    index_count  = indices.size();
    vertex_count = vertices.size();

    indexBuffer = createBuffer(allocator, indices.size() * sizeof(uint32_t),
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                               VMA_MEMORY_USAGE_GPU_ONLY, indexBufferMemory);

    vertexBuffer
        = createBuffer(allocator, vertices.size() * sizeof(Vertex),
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                       VMA_MEMORY_USAGE_GPU_ONLY, vertexBufferMemory);

    auto physicalDevice = render.physicalDevice;
    auto device         = render.device;
    auto commandPool    = render.commandPool;
    auto queue          = render.graphicsQueue;

    void *         data;
    VkBuffer       stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VkDeviceSize   bufferSize
        = std::fmax(vertices.size() * sizeof(Vertex), indices.size() * sizeof(uint32_t));
    stagingBuffer
        = createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       stagingBufferMemory);

    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), vertices.size() * sizeof(Vertex));
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);
    copyBuffer(commandBuffer, stagingBuffer, vertexBuffer, vertices.size() * sizeof(Vertex));
    endSingleTimeCommands(device, commandPool, queue, commandBuffer);

    memcpy(data, indices.data(), indices.size() * sizeof(uint32_t));
    commandBuffer = beginSingleTimeCommands(device, commandPool);
    copyBuffer(commandBuffer, stagingBuffer, indexBuffer, indices.size() * sizeof(uint32_t));
    endSingleTimeCommands(device, commandPool, queue, commandBuffer);
    vkUnmapMemory(device, stagingBufferMemory);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

MeshRender::MeshRender(MeshRender &&other) :
    renderer(other.renderer),
    raster(other.raster),
    topology(other.topology),
    index_count(other.index_count),
    vertex_count(other.vertex_count),
    bound(other.bound) {
    indexBuffer        = other.indexBuffer;
    indexBufferMemory  = other.indexBufferMemory;
    vertexBuffer       = other.vertexBuffer;
    vertexBufferMemory = other.vertexBufferMemory;
    pso                = other.pso;
    name               = std::move(other.name);

    other.indexBuffer        = VK_NULL_HANDLE;
    other.indexBufferMemory  = VK_NULL_HANDLE;
    other.vertexBuffer       = VK_NULL_HANDLE;
    other.vertexBufferMemory = VK_NULL_HANDLE;
}

void MeshRender::render(uint32_t index) {
    auto pipeline       = renderer.acquirePipeline(ghostRaster, topology, vertShader, fragShader);
    auto pipelineLayout = renderer.pipelineLayout;
    auto commandBuffer  = renderer.commandBuffers[index];
    auto descriptorSet  = renderer.descriptorSets[index];

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);

    VkBuffer     vertexBuffers[] = {vertexBuffer};
    VkDeviceSize offsets[]       = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    if (raster == RasterizeType::WIREFRAME) {
        vkCmdSetLineWidth(commandBuffer, lineWidth);
    }

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                            &descriptorSet, 0, nullptr);

    vkCmdPushConstants(commandBuffer, pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(Render::PushConstantObject), &ghostPso);

    vkCmdDrawIndexed(commandBuffer, index_count, 1, 0, 0, 0);
}
