#pragma once

#include "config.h"
#include "utility.hpp"
#include "viewer.h"

#include "cluster.hpp"

#include <array>
#include <chrono>
#include <glm/gtx/hash.hpp>
#include <iostream>
#include <limits>
#include <span>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

struct Vertex {
    Vec3 pos;
    Vec3 normal;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding   = 0;
        bindingDescription.stride    = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding  = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset   = offsetof(Vertex, pos);

        attributeDescriptions[1].binding  = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset   = offsetof(Vertex, normal);

        return attributeDescriptions;
    }

    bool operator==(const Vertex &other) const {
        return pos == other.pos && normal == other.normal;
    }
};

static inline void hash_combine(size_t &seed, size_t hash) {
    hash += 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= hash;
}

struct VertexHasher {
    size_t operator()(Vertex const &vertex) const {
        size_t seed = 0;
        hash_combine(seed, PositionHasher()(vertex.pos));
        hash_combine(seed, PositionHasher()(vertex.normal));
        return seed;
    }
};

enum class TopologyType {
    POINT    = 0,
    LINE     = 1,
    TRIANGLE = 2,
};

enum class RasterizeType {
    FILLED    = 0,
    WIREFRAME = 1,
    POINT     = 2,
};

enum class ClusterColor {
    RED     = 0,
    GREEN   = 1,
    BLUE    = 2,
    YELLOW  = 3,
    MAGENTA = 4,
    CYAN    = 5,
    WHITE   = 6,
};

struct Render {
    static constexpr uint32_t maxUboCount   = 1000U;
    static constexpr uint32_t maxImageCount = 100U;

    using PipelineKey = std::tuple<RasterizeType, TopologyType, std::string, std::string>;
    struct PipelineKeyHasher {
        size_t operator()(const PipelineKey &key) const {
            size_t seed = 0;
            hash_combine(seed, std::hash<RasterizeType>()(std::get<0>(key)));
            hash_combine(seed, std::hash<TopologyType>()(std::get<1>(key)));
            hash_combine(seed, std::hash<std::string>()(std::get<2>(key)));
            hash_combine(seed, std::hash<std::string>()(std::get<3>(key)));
            return seed;
        }
    };

    struct Pipeline {
        std::string vertShader;
        std::string fragShader;
        VkPipeline  pipeline;
    };

    std::unordered_map<PipelineKey, Pipeline, PipelineKeyHasher> pipelines;

    struct UniformBufferObject {
        alignas(16) glm::mat4 model;
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
        alignas(16) glm::vec4 lightPos;
        alignas(16) glm::vec4 lightDir;
        alignas(16) glm::vec4 lightColor;
    } ubo;

    struct PushConstantObject {
        alignas(16) uint32_t clusterColorId = 0U;
        alignas(16) glm::mat4 model         = glm::mat4(1.0f);
    };

    std::string title;

    GLFWwindow *          window = nullptr;
    VkInstance            instance;
    VkSurfaceKHR          surface;
    VkPhysicalDevice      physicalDevice;
    VkDevice              device;
    VkQueue               graphicsQueue;
    VkCommandPool         commandPool;
    VkSwapchainKHR        swapChain;
    VkRenderPass          renderPass;
    VkDescriptorPool      descriptorPool;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout      pipelineLayout;

    VmaAllocator allocator;

    uint32_t currentBuffer = 0U;
    uint32_t imageIndex    = 0U;
    float    fps           = 0.0f;
    float    ms            = 0.0f;
    float    dt            = 0.0f;
    uint64_t frameCount    = 0U;

    std::array<VkCommandBuffer, MAX_IN_FLIGHT_FRAMES> commandBuffers;
    std::array<VkSemaphore, MAX_IN_FLIGHT_FRAMES>     imageAvailableSemaphores;
    std::array<VkSemaphore, MAX_IN_FLIGHT_FRAMES>     renderFinishedSemaphores;
    std::array<VkFence, MAX_IN_FLIGHT_FRAMES>         inFlightFences;
    std::array<VkDescriptorSet, MAX_IN_FLIGHT_FRAMES> descriptorSets;
    std::array<VkBuffer, MAX_IN_FLIGHT_FRAMES>        uniformBuffers;
    std::array<VmaAllocation, MAX_IN_FLIGHT_FRAMES>   uniformBufferMemorys;
    std::array<void *, MAX_IN_FLIGHT_FRAMES>          uniformBufferMapped;

    VkExtent2D                  swapchainExtent;
    VkSurfaceFormatKHR          surfaceFormat;
    uint32_t                    swapchainImageCount;
    std::vector<VkImage>        swapchainImages;
    std::vector<VkImageView>    swapchainImageViews;
    VkFormat                    depthFormat;
    std::vector<VkImage>        depthImages;
    std::vector<VkDeviceMemory> depthImageMemories;
    std::vector<VkImageView>    depthImageViews;
    std::vector<VkFramebuffer>  framebuffers;

    VkDescriptorPool imGuiDescriptorPool;

    Render(std::string_view title);

    Pipeline acquirePipeline(RasterizeType raster, TopologyType topology, std::string vertShader,
                             std::string fragShader);

    ~Render();

    void updateUbo();

    using RenderFunc = std::function<void()>;
    int loop(RenderFunc preProcess, RenderFunc render, RenderFunc postProcess);
};

struct MeshRender {
    static constexpr auto maxRenderCount = 5000;
    static constexpr auto vertShader     = FILESYSTEM_ROOT "/build/shaders/cluster.vert.spv";
    static constexpr auto fragShader     = FILESYSTEM_ROOT "/build/shaders/cluster.frag.spv";

    VkBuffer      vertexBuffer;
    VmaAllocation vertexBufferMemory;
    VkBuffer      indexBuffer;
    VmaAllocation indexBufferMemory;

    Render::PushConstantObject pso{static_cast<uint32_t>(ClusterColor::WHITE)};
    Render::PushConstantObject ghostPso{static_cast<uint32_t>(ClusterColor::WHITE)};

    Render &renderer;

    RasterizeType raster;
    RasterizeType ghostRaster;
    TopologyType  topology;
    uint32_t      index_count;
    uint32_t      vertex_count;
    Bounds        bound;

    float lineWidth = 1.0f;

    std::string name;

    MeshRender(std::string name, Render &render, std::span<Vertex> vertices,
               std::span<uint32_t> indices, TopologyType topology, RasterizeType raster);

    MeshRender(MeshRender &&other);

    ~MeshRender() {
        if (indexBufferMemory) {
            auto allocator = renderer.allocator;
            vmaDestroyBuffer(allocator, indexBuffer, indexBufferMemory);
            vmaDestroyBuffer(allocator, vertexBuffer, vertexBufferMemory);
        }
    }

    void render(uint32_t index);
};

void mesh_gui(MeshRender &instance) noexcept;

struct ClusterData {
    std::vector<Vec3>     vertices;
    std::vector<uint32_t> indices;
};

struct Scene {
    std::vector<MeshRender>  meshes;
    std::vector<MeshRender>  boundingBoxes;
    std::vector<ClusterData> clusters;
    enum Mask {
        enabled = 0x01,
        bounded = 0x02,
    };
    std::vector<uint8_t> flags;
};

void emplace_mesh(Scene &scene, Render &render, std::string name, std::span<Vertex> vertices,
                  std::span<uint32_t> indices, TopologyType topology, RasterizeType raster);

void scene_gui(Scene &scene) noexcept;

uint32_t ray_cast_nearest(Scene &scene, Vec3 ori, Vec3 dir, float &t,
                          float tmin = std::numeric_limits<float>::lowest(),
                          float tmax = std::numeric_limits<float>::max());
