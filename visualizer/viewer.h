#pragma once

#include "config.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <glm/glm.hpp>

#include <vk_mem_alloc.h>

static constexpr int MAX_IN_FLIGHT_FRAMES = 2;
static constexpr int WINDOW_WIDTH         = 1500;
static constexpr int WINDOW_HEIGHT        = 960;

extern bool  c_invert_y;
extern float c_fov;
extern float c_far;
extern float c_near;
extern float sensitivity;
extern float speed;
extern float speed_up;

extern float     smooth_factor;
extern glm::vec3 g_velocity;
extern glm::vec3 g_smoothedVelocity;

extern float lastX;
extern float lastY;
extern float yaw;
extern float pitch;

extern glm::vec3 c_front;
extern glm::vec3 c_right;
extern glm::vec3 c_up_p;
extern glm::vec3 c_pos;

extern glm::vec4 g_light_pos;
extern glm::vec4 g_light_col;
extern glm::vec4 g_light_dir;

extern glm::mat4 g_view;
extern glm::mat4 g_proj;

template <std::floating_point T>
static inline T lerp(T a, T b, T f) {
    return a + f * (b - a);
}
glm::vec3 lerp(const glm::vec3 &a, const glm::vec3 &b, float f);
void      cursorPosCallback(GLFWwindow *window, double xpos, double ypos);
void      keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void      scrollCallback(GLFWwindow *window, double xoffset, double yoffset);
auto      eval_front(float yaw, float pitch) -> glm::vec3;
auto      eval_right(glm::vec3 front, float yaw, float pitch) -> glm::vec3;
auto      eval_up(glm::vec3 front, glm::vec3 right) -> glm::vec3;
auto      update_camera(float dt) -> void;
bool      isGuiHovered() noexcept;

void camera_gui() noexcept;
void light_gui() noexcept;

std::vector<char> readFile(const std::string &filename);

std::vector<const char *> getGlfwExtensions();
uint32_t                  selectQueueFamily(VkSurfaceKHR surface, VkPhysicalDevice device);
bool                      checkDeviceExtensionSupport(VkPhysicalDevice                 device,
                                                      const std::vector<const char *> &extensions);
bool                      isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface,
                                           const std::vector<const char *> &extensions);
VkShaderModule            createShaderModule(VkDevice device, const std::vector<char> &code);
VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats);
VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes);
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR        capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
};
SwapChainSupportDetails getSwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);
VkExtent2D       chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities, GLFWwindow *window);
VkInstance       createInstance(const std::vector<const char *> &extensions);
VkSurfaceKHR     createSurface(VkInstance instance, GLFWwindow *window);
VkPhysicalDevice pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface,
                                    const std::vector<const char *> &extensions);
VkDevice         createDevice(VkInstance instance, VkPhysicalDevice physicalDevice,
                              std::vector<uint32_t> &          queueFamilyIndices,
                              const std::vector<const char *> &extensions);
VmaAllocator createAllocator(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device);
void         destroyAllocator(VmaAllocator allocator);
VkSwapchainKHR   createSwapchain(VkPhysicalDevice physicalDevice, VkDevice device,
                                 VkSurfaceKHR surface, uint32_t queueFamilyIndex, VkExtent2D extent,
                                 VkSurfaceFormatKHR surfaceFormat, VkPresentModeKHR presentMode,
                                 VkSwapchainKHR oldSwapChain = VK_NULL_HANDLE);
VkQueue          getQueue(VkDevice device, uint32_t queueFamilyIndex);
VkCommandPool    createCommandPool(VkDevice device, uint32_t queueFamilyIndex);
VkCommandBuffer  createCommandBuffer(VkDevice device, VkCommandPool commandPool);
VkCommandBuffer  beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool);
void             endSingleTimeCommands(VkDevice device, VkCommandPool commandPool, VkQueue queue,
                                       VkCommandBuffer commandBuffer);
VkSemaphore      createSemaphore(VkDevice device);
VkFence          createFence(VkDevice device);
uint32_t         findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                                VkMemoryPropertyFlags properties);
VkBuffer         createBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkDeviceSize size,
                              VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                              VkDeviceMemory &bufferMemory);
VkBuffer         createBuffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage,
                              VmaMemoryUsage memoryUsage, VmaAllocation &allocation);
VkImage          createImage(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t width,
                             uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
                             VkMemoryPropertyFlags properties, VkDeviceMemory &imageMemory);
VkImageView      createImageView(VkDevice device, VkImage image, VkFormat format,
                                 VkImageAspectFlags aspectFlags);
VkFormat         pickDepthFormat(VkPhysicalDevice physicalDevice);
VkRenderPass     createRenderPass(VkDevice device, VkFormat colorFormat, VkFormat depthFormat);
VkFramebuffer    createFramebuffer(VkDevice device, VkRenderPass renderPass, VkImageView imageView,
                                   VkImageView depthImageView, VkExtent2D extent);
VkDescriptorPool createDescriptorPool(VkDevice device, uint32_t uniformBufferCount,
                                      uint32_t imageSamplerCount, uint32_t storageBufferCount = 0,
                                      uint32_t storageImageCount = 0);
void transitionImageLayout(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkImage image,
                           VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
void copyBuffer(VkCommandBuffer cmdBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer,
                VkDeviceSize size);

// glfw window utility
GLFWwindow *createWindow(std::string title, int width, int height);
void        cleanupWindow(GLFWwindow *window);

// imgui utility
void createImGuiContext(GLFWwindow *window, VkInstance instance, VkPhysicalDevice physicalDevice,
                        VkDevice device, VkQueue queue, uint32_t queueFamilyIndex,
                        VkCommandPool commandPool, VkDescriptorPool descriptorPool,
                        VkSurfaceKHR surface, VkRenderPass renderPass, uint32_t imageCount);
void newImGuiFrame();
void endImGuiFrame();
void renderImGui(VkCommandBuffer commandBuffer);
void cleanupImGuiContext();

// render utility
struct BoxRender {
    static constexpr auto vertShader = FILESYSTEM_ROOT "/build/shaders/box.vert.spv";
    static constexpr auto fragShader = FILESYSTEM_ROOT "/build/shaders/box.frag.spv";

    static constexpr float vertices[8 * 3] = {
        -1.0f, -1.0f, -1.0f, // 0
        -1.0f, -1.0f, 1.0f,  // 1
        -1.0f, 1.0f,  1.0f,  // 2
        -1.0f, 1.0f,  -1.0f, // 3
        1.0f,  -1.0f, -1.0f, // 4
        1.0f,  -1.0f, 1.0f,  // 5
        1.0f,  1.0f,  1.0f,  // 6
        1.0f,  1.0f,  -1.0f, // 7
    };

    static constexpr uint32_t lineIndex[] = {
        0, 1, 1, 2, 2, 3, 3, 0, // bottom
        4, 5, 5, 6, 6, 7, 7, 4, // top
        0, 4, 1, 5, 2, 6, 3, 7, // sides
    };

    struct UniformBufferObject {
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
    } ubo;

    float lineWidth = 1.0f;

    std::vector<glm::vec4> modelMat;

    std::vector<VkBuffer>        instanceBuffers;
    std::vector<VkDeviceMemory>  instanceBufferMemorys;
    std::vector<VkBuffer>        uniformBuffers;
    std::vector<VkDeviceMemory>  uniformBufferMemorys;
    std::vector<VkDescriptorSet> descriptorSets;

    static VkBuffer              vertexBuffer;
    static VkDeviceMemory        vertexBufferMemory;
    static VkBuffer              indexBuffer;
    static VkDeviceMemory        indexBufferMemory;
    static VkDescriptorSetLayout descriptorSetLayout;
    static VkDescriptorPool      descriptorPool;
    static VkPipelineLayout      pipelineLayout;
    static VkPipeline            pipeline;
};
void      initBoxRender(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool,
                        VkQueue queue, VkRenderPass renderPass);
BoxRender createBoxRenderInstance(VkPhysicalDevice physicalDevice, VkDevice device,
                                  VkCommandPool commandPool, VkQueue queue,
                                  const std::vector<glm::vec3> &pos,
                                  const std::vector<glm::vec3> &scale,
                                  const std::vector<glm::vec3> &ori);
void      renderBoxes(VkCommandBuffer commandBuffer, const BoxRender &instance);
void      updateBoxUbo(VkDevice device, BoxRender &instance, const glm::mat4 &view,
                       const glm::mat4 &proj);
void      destroyBoxRendererInstance(VkDevice device, BoxRender renderer);
void      cleanupBoxRenderer(VkDevice device);
