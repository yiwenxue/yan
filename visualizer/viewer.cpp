#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include "viewer.h"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <imgui.h>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan_core.h>

bool  c_invert_y  = true;  // invert y axis for camera contorl or not
float c_fov       = 60.0f; // field of view
float c_near      = 0.01f;
float c_far       = 100.0f;
float sensitivity = 0.1f; // mouse sensitivity
float speed       = 0.5f; // camera move speed
float speed_up    = 5.0;  // speed up when shift pressed

float     smooth_factor      = 0.2; // larger, high response
glm::vec3 g_velocity         = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 g_smoothedVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

float lastX = 0.0f;
float lastY = 0.0f;
float yaw   = 270.0f;
float pitch = -10.0f;

glm::vec3 c_front = eval_front(yaw, pitch);
glm::vec3 c_right = eval_right(c_front, yaw, pitch);
glm::vec3 c_up_p  = eval_up(c_front, c_right);
glm::vec3 c_pos   = glm::vec3(0.0f, 0.5f, 2.0f);

glm::vec4 g_light_pos = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
glm::vec4 g_light_dir = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
glm::vec4 g_light_col = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

glm::mat4 g_view = glm::lookAt(c_pos, c_pos + c_front, c_up_p);
glm::mat4 g_proj = glm::perspective(glm::radians(c_fov),
                                    (float) WINDOW_WIDTH / (float) WINDOW_HEIGHT, 0.1f, 100.0f);

std::vector<char> readFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t            fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

std::vector<const char *> getGlfwExtensions() {
    uint32_t     glfwExtensionCount = 0;
    const char **glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    return extensions;
}

uint32_t selectQueueFamily(VkSurfaceKHR surface, VkPhysicalDevice device) {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    uint32_t i = 0;
    for (const auto &queueFamily : queueFamilies) {
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT && presentSupport) {
            return i;
        }
        i++;
    }
    throw std::runtime_error("failed to find a suitable queue family!");
}

bool checkDeviceExtensionSupport(VkPhysicalDevice                 device,
                                 const std::vector<const char *> &extensions) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         availableExtensions.data());
    std::set<std::string> requiredExtensions(extensions.begin(), extensions.end());
    for (const auto &extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }
    return requiredExtensions.empty();
}

bool isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface,
                      const std::vector<const char *> &extensions) {
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
    bool extensionsSupported = checkDeviceExtensionSupport(device, extensions);
    bool swapChainAdequate   = false;
    if (extensionsSupported) {
        VkSurfaceCapabilitiesKHR capabilities;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &capabilities);
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
        swapChainAdequate = formatCount != 0 && presentModeCount != 0;
    }
    return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
           && deviceFeatures.geometryShader && extensionsSupported && swapChainAdequate;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char> &code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode    = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule;
}

// swapchain utils
VkSurfaceFormatKHR
    chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats) {
    for (const auto &availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB
            && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }
    return availableFormats[0];
}

VkPresentModeKHR presentModePriority[] = {
    VK_PRESENT_MODE_FIFO_KHR,
    VK_PRESENT_MODE_MAILBOX_KHR,
    VK_PRESENT_MODE_IMMEDIATE_KHR,
};

VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes) {
    for (auto mode : presentModePriority) {
        if (std::find(availablePresentModes.begin(), availablePresentModes.end(), mode)
            != availablePresentModes.end()) {
            return mode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities, GLFWwindow *window) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        int        width, height;
        VkExtent2D actualExtent;
        glfwGetFramebufferSize(window, &width, &height);
        actualExtent.width
            = std::clamp(static_cast<uint32_t>(width), capabilities.minImageExtent.width,
                         capabilities.maxImageExtent.width);
        actualExtent.height
            = std::clamp(static_cast<uint32_t>(height), capabilities.minImageExtent.height,
                         capabilities.maxImageExtent.height);
        return actualExtent;
    }
}

SwapChainSupportDetails getSwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface) {
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount,
                                                  details.presentModes.data());
    }
    return details;
}

// vulkan utils

VkInstance createInstance(const std::vector<const char *> &extensions) {
    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 3, 0);
    appInfo.pEngineName        = "No Engine";
    appInfo.engineVersion      = VK_MAKE_VERSION(1, 3, 0);
    appInfo.apiVersion         = VK_API_VERSION_1_3;
    VkInstanceCreateInfo createInfo{};
    createInfo.sType                 = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo      = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size()); // glfw extensions
    createInfo.ppEnabledExtensionNames = extensions.data();
    VkInstance instance;
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }
    return instance;
}

VkSurfaceKHR createSurface(VkInstance instance, GLFWwindow *window) {
    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
    return surface;
}

VkPhysicalDevice pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface,
                                    const std::vector<const char *> &extensions) {
    uint32_t         deviceCount = 0;
    VkPhysicalDevice device;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    for (const auto &device : devices) {
        if (isDeviceSuitable(device, surface, extensions)) {
            return device;
        }
    }
    throw std::runtime_error("failed to find a suitable GPU!");
}

VkDevice createDevice(VkInstance instance, VkPhysicalDevice physicalDevice,
                      std::vector<uint32_t> &          queueFamilyIndices,
                      const std::vector<const char *> &extensions) {
    float                                queuePriority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies(queueFamilyIndices.begin(), queueFamilyIndices.end());
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount       = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{.fillModeNonSolid = VK_TRUE, .wideLines = VK_TRUE};
    VkDeviceCreateInfo       createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos       = queueCreateInfos.data();
    createInfo.pEnabledFeatures        = &deviceFeatures;
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkDevice device;
    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }

    return device;
}

VmaAllocator createAllocator(VkInstance instance, VkPhysicalDevice physicalDevice,
                             VkDevice device) {
    VmaVulkanFunctions vulkanFunctions    = {};
    vulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr   = &vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocatorCreateInfo = {};
    allocatorCreateInfo.vulkanApiVersion       = VK_API_VERSION_1_2;
    allocatorCreateInfo.physicalDevice         = physicalDevice;
    allocatorCreateInfo.device                 = device;
    allocatorCreateInfo.instance               = instance;
    allocatorCreateInfo.pVulkanFunctions       = &vulkanFunctions;

    VmaAllocator allocator;
    if (vmaCreateAllocator(&allocatorCreateInfo, &allocator) != VK_SUCCESS) {
        throw std::runtime_error("failed to create allocator!");
    }

    return allocator;
}

void destroyAllocator(VmaAllocator allocator) {
    vmaDestroyAllocator(allocator);
}

VkSwapchainKHR createSwapchain(VkPhysicalDevice physicalDevice, VkDevice device,
                               VkSurfaceKHR surface, uint32_t queueFamilyIndex, VkExtent2D extent,
                               VkSurfaceFormatKHR surfaceFormat, VkPresentModeKHR presentMode,
                               VkSwapchainKHR oldSwapChain) {
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities);
    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
        imageCount = capabilities.maxImageCount;
    }
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType                 = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface               = surface;
    createInfo.minImageCount         = imageCount;
    createInfo.imageFormat           = surfaceFormat.format;
    createInfo.imageColorSpace       = surfaceFormat.colorSpace;
    createInfo.imageExtent           = extent;
    createInfo.imageArrayLayers      = 1;
    createInfo.imageUsage            = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 1;
    createInfo.pQueueFamilyIndices   = &queueFamilyIndex;
    createInfo.preTransform          = capabilities.currentTransform;
    createInfo.compositeAlpha        = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode           = presentMode;
    createInfo.clipped               = VK_TRUE;
    createInfo.oldSwapchain          = oldSwapChain;
    VkSwapchainKHR swapChain;
    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }
    return swapChain;
}

VkQueue getQueue(VkDevice device, uint32_t queueFamilyIndex) {
    VkQueue queue;
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
    return queue;
}

VkCommandPool createCommandPool(VkDevice device, uint32_t queueFamilyIndex) {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool commandPool;
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
    return commandPool;
}

VkCommandBuffer createCommandBuffer(VkDevice device, VkCommandPool commandPool) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = commandPool;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
    return commandBuffer;
}

VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool        = commandPool;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    return commandBuffer;
}

void endSingleTimeCommands(VkDevice device, VkCommandPool commandPool, VkQueue queue,
                           VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);
    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &commandBuffer;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

VkSemaphore createSemaphore(VkDevice device) {
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkSemaphore semaphore;
    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS) {
        throw std::runtime_error("failed to create semaphores!");
    }
    return semaphore;
}

VkFence createFence(VkDevice device) {
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkFence fence;
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        throw std::runtime_error("failed to create fences!");
    }
    return fence;
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                        VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i))
            && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

VkBuffer createBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkDeviceSize size,
                      VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                      VkDeviceMemory &bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size        = size;
    bufferInfo.usage       = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create vertex buffer!");
    }
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex
        = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate vertex buffer memory!");
    }
    vkBindBufferMemory(device, buffer, bufferMemory, 0);
    return buffer;
}

VkBuffer createBuffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage,
                      VmaMemoryUsage memoryUsage, VmaAllocation &allocation) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size        = size;
    bufferInfo.usage       = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memoryUsage;

    VkBuffer buffer;
    if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr)
        != VK_SUCCESS) {
        throw std::runtime_error("failed to create vertex buffer!");
    }
    return buffer;
}

VkImage createImage(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t width,
                    uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkDeviceMemory &imageMemory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width  = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth  = 1;
    imageInfo.mipLevels     = 1;
    imageInfo.arrayLayers   = 1;
    imageInfo.format        = format;
    imageInfo.tiling        = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage         = usage;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    VkImage image;
    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex
        = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }
    vkBindImageMemory(device, image, imageMemory, 0);
    return image;
}

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format,
                            VkImageAspectFlags aspectFlags) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = image;
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format                          = format;
    viewInfo.subresourceRange.aspectMask     = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;
    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }
    return imageView;
}

void copyBuffer(VkCommandBuffer cmdBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer,
                VkDeviceSize size) {
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(cmdBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
}

VkFormat pickDepthFormat(VkPhysicalDevice physicalDevice) {
    std::vector<VkFormat> candidates = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT,
    };
    for (auto format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
        if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            return format;
        }
    }
    throw std::runtime_error("failed to find supported format!");
}

VkRenderPass createRenderPass(VkDevice device, VkFormat swapChainImageFormat,
                              VkFormat depthFormat) {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format         = swapChainImageFormat;
    colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentDescription depthAttachment{};
    depthAttachment.format         = depthFormat;
    depthAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments    = attachments.data();
    renderPassInfo.subpassCount    = 1;
    renderPassInfo.pSubpasses      = &subpass;
    VkRenderPass renderPass;
    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
    return renderPass;
}

VkDescriptorPool createDescriptorPool(VkDevice device, uint32_t uniformBufferCount,
                                      uint32_t imageSamplerCount, uint32_t storageBufferCount,
                                      uint32_t storageImageCount) {
    std::vector<VkDescriptorPoolSize> poolSizes;
    if (uniformBufferCount > 0) {
        poolSizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uniformBufferCount});
    }
    if (imageSamplerCount > 0) {
        poolSizes.push_back({VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageSamplerCount});
    }
    if (storageBufferCount > 0) {
        poolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, storageBufferCount});
    }
    if (storageImageCount > 0) {
        poolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageImageCount});
    }
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes    = poolSizes.data();
    poolInfo.maxSets
        = uniformBufferCount + imageSamplerCount + storageBufferCount + storageImageCount;
    VkDescriptorPool descriptorPool;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
    return descriptorPool;
}

VkFramebuffer createFramebuffer(VkDevice device, VkRenderPass renderPass, VkImageView imageView,
                                VkImageView depthImageView, VkExtent2D extent) {
    std::array<VkImageView, 2> attachments = {imageView, depthImageView};
    VkFramebufferCreateInfo    framebufferInfo{};
    framebufferInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass      = renderPass;
    framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    framebufferInfo.pAttachments    = attachments.data();
    framebufferInfo.width           = extent.width;
    framebufferInfo.height          = extent.height;
    framebufferInfo.layers          = 1;
    VkFramebuffer framebuffer;
    if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
    }
    return framebuffer;
}

void transitionImageLayout(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkImage image,
                           VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBuffer      commandBuffer = beginSingleTimeCommands(device, commandPool);
    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = oldLayout;
    barrier.newLayout           = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = image;
    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT) {
            barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
    } else {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = 1;
    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
        && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage           = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
               && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage           = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage      = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
               && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                                | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        sourceStage      = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
               && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask
            = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        sourceStage      = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    } else {
        throw std::invalid_argument("unsupported layout transition!");
    }
    vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1,
                         &barrier);
    endSingleTimeCommands(device, commandPool, queue, commandBuffer);
}

GLFWwindow *createWindow(std::string title, int width, int height) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    double      xd, yd;
    glfwGetCursorPos(window, &xd, &yd);
    lastX = xd;
    lastY = yd;
    return window;
}

void cleanupWindow(GLFWwindow *window) {
    glfwDestroyWindow(window);
    glfwTerminate();
}

// create imgui vulkan context
void createImGuiContext(GLFWwindow *window, VkInstance instance, VkPhysicalDevice physicalDevice,
                        VkDevice device, VkQueue queue, uint32_t queueFamilyIndex,
                        VkCommandPool commandPool, VkDescriptorPool descriptorPool,
                        VkSurfaceKHR surface, VkRenderPass renderPass, uint32_t imageCount) {
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance        = instance;
    initInfo.PhysicalDevice  = physicalDevice;
    initInfo.Device          = device;
    initInfo.QueueFamily     = queueFamilyIndex;
    initInfo.Queue           = queue;
    initInfo.PipelineCache   = VK_NULL_HANDLE;
    initInfo.DescriptorPool  = descriptorPool;
    initInfo.Allocator       = nullptr;
    initInfo.MinImageCount   = imageCount;
    initInfo.ImageCount      = imageCount;
    initInfo.CheckVkResultFn = nullptr;
    ImGui_ImplVulkan_Init(&initInfo, renderPass);
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);
    ImGui_ImplVulkan_CreateFontsTexture(commandBuffer);
    endSingleTimeCommands(device, commandPool, queue, commandBuffer);
    ImGui_ImplVulkan_DestroyFontUploadObjects();
}

// new frame
void newImGuiFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void endImGuiFrame() {
    ImGui::Render();
}

void renderImGui(VkCommandBuffer commandBuffer) {
    ImDrawData *drawData = ImGui::GetDrawData();

    // if no gui recorded, skip
    if (!drawData || drawData->TotalVtxCount == 0) {
        return;
    }

    ImGui_ImplVulkan_RenderDrawData(drawData, commandBuffer);
}

void cleanupImGuiContext() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

VkBuffer              BoxRender::vertexBuffer        = VK_NULL_HANDLE;
VkDeviceMemory        BoxRender::vertexBufferMemory  = VK_NULL_HANDLE;
VkBuffer              BoxRender::indexBuffer         = VK_NULL_HANDLE;
VkDeviceMemory        BoxRender::indexBufferMemory   = VK_NULL_HANDLE;
VkDescriptorSetLayout BoxRender::descriptorSetLayout = VK_NULL_HANDLE;
VkDescriptorPool      BoxRender::descriptorPool      = VK_NULL_HANDLE;
VkPipelineLayout      BoxRender::pipelineLayout      = VK_NULL_HANDLE;
VkPipeline            BoxRender::pipeline            = VK_NULL_HANDLE;

void initBoxRender(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool,
                   VkQueue queue, VkRenderPass renderPass) {
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding         = 0;
    uboLayoutBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo uboLayoutInfo{};
    uboLayoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    uboLayoutInfo.bindingCount = 1;
    uboLayoutInfo.pBindings    = &uboLayoutBinding;
    if (vkCreateDescriptorSetLayout(device, &uboLayoutInfo, nullptr,
                                    &BoxRender::descriptorSetLayout)
        != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = MAX_IN_FLIGHT_FRAMES;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = &poolSize;
    poolInfo.maxSets       = MAX_IN_FLIGHT_FRAMES;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &BoxRender::descriptorPool)
        != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }

    BoxRender::vertexBuffer
        = createBuffer(physicalDevice, device, sizeof(BoxRender::vertices),
                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, BoxRender::vertexBufferMemory);
    BoxRender::indexBuffer
        = createBuffer(physicalDevice, device, sizeof(BoxRender::lineIndex),
                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, BoxRender::indexBufferMemory);

    void *         data;
    VkBuffer       stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VkDeviceSize stagingSize = std::max(sizeof(BoxRender::lineIndex), sizeof(BoxRender::vertices));
    stagingBuffer
        = createBuffer(physicalDevice, device, stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       stagingBufferMemory);

    VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);
    vkMapMemory(device, stagingBufferMemory, 0, sizeof(BoxRender::vertices), 0, &data);
    memcpy(data, BoxRender::vertices, sizeof(BoxRender::vertices));
    vkUnmapMemory(device, stagingBufferMemory);
    copyBuffer(commandBuffer, stagingBuffer, BoxRender::vertexBuffer, sizeof(BoxRender::vertices));
    endSingleTimeCommands(device, commandPool, queue, commandBuffer);

    commandBuffer = beginSingleTimeCommands(device, commandPool);
    vkMapMemory(device, stagingBufferMemory, 0, sizeof(BoxRender::lineIndex), 0, &data);
    memcpy(data, BoxRender::lineIndex, sizeof(BoxRender::lineIndex));
    vkUnmapMemory(device, stagingBufferMemory);
    copyBuffer(commandBuffer, stagingBuffer, BoxRender::indexBuffer, sizeof(BoxRender::lineIndex));
    endSingleTimeCommands(device, commandPool, queue, commandBuffer);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    VkVertexInputBindingDescription bindingDescription[2] = {};
    bindingDescription[0].binding                         = 0;
    bindingDescription[0].stride                          = 3 * sizeof(float);
    bindingDescription[0].inputRate                       = VK_VERTEX_INPUT_RATE_VERTEX;
    bindingDescription[1].binding                         = 1;
    bindingDescription[1].stride                          = 3 * sizeof(glm::vec4);
    bindingDescription[1].inputRate                       = VK_VERTEX_INPUT_RATE_INSTANCE;

    VkVertexInputAttributeDescription attributeDescriptions[4] = {};
    attributeDescriptions[0].binding                           = 0;
    attributeDescriptions[0].location                          = 0;
    attributeDescriptions[0].format                            = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset                            = 0;
    attributeDescriptions[1].binding                           = 1;
    attributeDescriptions[1].location                          = 1;
    attributeDescriptions[1].format                            = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[1].offset                            = 0;
    attributeDescriptions[2].binding                           = 1;
    attributeDescriptions[2].location                          = 2;
    attributeDescriptions[2].format                            = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[2].offset                            = sizeof(glm::vec4);
    attributeDescriptions[3].binding                           = 1;
    attributeDescriptions[3].location                          = 3;
    attributeDescriptions[3].format                            = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[3].offset                            = 2 * sizeof(glm::vec4);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount   = 2;
    vertexInputInfo.pVertexBindingDescriptions      = bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = 4;
    vertexInputInfo.pVertexAttributeDescriptions    = attributeDescriptions;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = (float) WINDOW_WIDTH;
    viewport.height   = (float) WINDOW_HEIGHT;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {WINDOW_WIDTH, WINDOW_HEIGHT};

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
    rasterizer.polygonMode             = VK_POLYGON_MODE_LINE;
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

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts    = &BoxRender::descriptorSetLayout;
    if (vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr,
                               &BoxRender::pipelineLayout)
        != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkDynamicState dynamicStates[]
        = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_LINE_WIDTH};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 3;
    dynamicState.pDynamicStates    = dynamicStates;

    const auto vertCode = readFile(BoxRender::vertShader);
    const auto fragCode = readFile(BoxRender::fragShader);

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
    pipelineInfo.layout              = BoxRender::pipelineLayout;
    pipelineInfo.renderPass          = renderPass;
    pipelineInfo.subpass             = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                  &BoxRender::pipeline)
        != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

BoxRender createBoxRenderInstance(VkPhysicalDevice physicalDevice, VkDevice device,
                                  VkCommandPool commandPool, VkQueue queue,
                                  const std::vector<glm::vec3> &pos,
                                  const std::vector<glm::vec3> &scale,
                                  const std::vector<glm::vec3> &ori) {
    BoxRender boxRender{};
    boxRender.ubo = {
        glm::mat4(1.0f),
        glm::mat4(1.0f),
    };

    boxRender.uniformBuffers.resize(MAX_IN_FLIGHT_FRAMES);
    boxRender.uniformBufferMemorys.resize(MAX_IN_FLIGHT_FRAMES);
    for (size_t i = 0; i < MAX_IN_FLIGHT_FRAMES; ++i) {
        boxRender.uniformBuffers[i] = createBuffer(
            physicalDevice, device, sizeof(BoxRender::UniformBufferObject),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            boxRender.uniformBufferMemorys[i]);
    }

    boxRender.descriptorSets.resize(MAX_IN_FLIGHT_FRAMES);
    std::vector<VkDescriptorSetLayout> layouts(MAX_IN_FLIGHT_FRAMES,
                                               BoxRender::descriptorSetLayout);
    VkDescriptorSetAllocateInfo        allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = BoxRender::descriptorPool;
    allocInfo.descriptorSetCount = MAX_IN_FLIGHT_FRAMES;
    allocInfo.pSetLayouts        = layouts.data();
    if (vkAllocateDescriptorSets(device, &allocInfo, boxRender.descriptorSets.data())
        != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    // update descriptor set
    for (size_t i = 0; i < MAX_IN_FLIGHT_FRAMES; ++i) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = boxRender.uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range  = sizeof(BoxRender::UniformBufferObject);

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet           = boxRender.descriptorSets[i];
        descriptorWrite.dstBinding       = 0;
        descriptorWrite.dstArrayElement  = 0;
        descriptorWrite.descriptorType   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.descriptorCount  = 1;
        descriptorWrite.pBufferInfo      = &bufferInfo;
        descriptorWrite.pImageInfo       = nullptr; // Optional
        descriptorWrite.pTexelBufferView = nullptr; // Optional

        vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
    }

    boxRender.modelMat.resize(3 * pos.size());

    for (size_t i = 0; i < pos.size(); ++i) {
        glm::mat4 model = glm::mat4(1.0f);
        model           = glm::translate(model, pos[i]);
        model           = glm::scale(model, scale[i]);
        model           = glm::rotate(model, glm::radians(ori[i].x), glm::vec3(1.0f, 0.0f, 0.0f));
        model           = glm::rotate(model, glm::radians(ori[i].y), glm::vec3(0.0f, 1.0f, 0.0f));
        model           = glm::rotate(model, glm::radians(ori[i].z), glm::vec3(0.0f, 0.0f, 1.0f));
        model           = glm::transpose(model);
        boxRender.modelMat[i * 3]     = model[0];
        boxRender.modelMat[i * 3 + 1] = model[1];
        boxRender.modelMat[i * 3 + 2] = model[2];
    }

    boxRender.instanceBuffers.resize(MAX_IN_FLIGHT_FRAMES);
    boxRender.instanceBufferMemorys.resize(MAX_IN_FLIGHT_FRAMES);
    VkBuffer       stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VkDeviceSize   stagingSize = sizeof(glm::vec4) * 3 * pos.size();
    stagingBuffer
        = createBuffer(physicalDevice, device, stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       stagingBufferMemory);

    for (size_t i = 0; i < MAX_IN_FLIGHT_FRAMES; ++i) {
        boxRender.instanceBuffers[i]
            = createBuffer(physicalDevice, device, sizeof(glm::vec4) * 3 * pos.size(),
                           VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, boxRender.instanceBufferMemorys[i]);

        void *data;
        vkMapMemory(device, stagingBufferMemory, 0, stagingSize, 0, &data);
        memcpy(data, boxRender.modelMat.data(), 3 * sizeof(glm::vec4) * pos.size());
        vkUnmapMemory(device, stagingBufferMemory);

        VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);
        copyBuffer(commandBuffer, stagingBuffer, boxRender.instanceBuffers[i], stagingSize);
        endSingleTimeCommands(device, commandPool, queue, commandBuffer);
    }

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    return std::move(boxRender);
}

void renderBoxes(VkCommandBuffer commandBuffer, const BoxRender &instance) {
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, BoxRender::pipeline);

    vkCmdSetLineWidth(commandBuffer, instance.lineWidth);

    const auto instance_count = instance.modelMat.size() / 3;

    VkBuffer     vertexBuffers[] = {BoxRender::vertexBuffer, instance.instanceBuffers[0]};
    VkDeviceSize offsets[]       = {0, 0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, BoxRender::indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            BoxRender::pipelineLayout, 0, 1, &instance.descriptorSets[0], 0,
                            nullptr);

    vkCmdDrawIndexed(commandBuffer, 24, instance_count, 0, 0, 0);
}

void updateBoxUbo(VkDevice device, BoxRender &instance, const glm::mat4 &view,
                  const glm::mat4 &proj) {
    instance.ubo.view = view;
    instance.ubo.proj = proj;

    void *data;
    vkMapMemory(device, instance.uniformBufferMemorys[0], 0, sizeof(BoxRender::UniformBufferObject),
                0, &data);
    memcpy(data, &instance.ubo, sizeof(BoxRender::UniformBufferObject));
    vkUnmapMemory(device, instance.uniformBufferMemorys[0]);
}

void destroyBoxRendererInstance(VkDevice device, BoxRender renderer) {
    for (size_t i = 0; i < MAX_IN_FLIGHT_FRAMES; ++i) {
        vkDestroyBuffer(device, renderer.instanceBuffers[i], nullptr);
        vkFreeMemory(device, renderer.instanceBufferMemorys[i], nullptr);
        vkDestroyBuffer(device, renderer.uniformBuffers[i], nullptr);
        vkFreeMemory(device, renderer.uniformBufferMemorys[i], nullptr);
    }
}

void cleanupBoxRenderer(VkDevice device) {
    vkDestroyDescriptorPool(device, BoxRender::descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, BoxRender::descriptorSetLayout, nullptr);
    vkDestroyPipeline(device, BoxRender::pipeline, nullptr);
    vkDestroyPipelineLayout(device, BoxRender::pipelineLayout, nullptr);
    vkDestroyBuffer(device, BoxRender::vertexBuffer, nullptr);
    vkFreeMemory(device, BoxRender::vertexBufferMemory, nullptr);
    vkDestroyBuffer(device, BoxRender::indexBuffer, nullptr);
    vkFreeMemory(device, BoxRender::indexBufferMemory, nullptr);
}

auto eval_front(float yaw, float pitch) -> glm::vec3 {
    float t_yaw   = yaw;
    float t_pitch = pitch;
    float up      = sin(glm::radians(t_pitch));
    float right   = cos(glm::radians(t_yaw)) * cos(glm::radians(t_pitch));
    float forward = sin(glm::radians(t_yaw)) * cos(glm::radians(t_pitch));
    return glm::normalize(glm::vec3(right, up, forward));
}

auto eval_right(glm::vec3 front, float yaw, float pitch) -> glm::vec3 {
    static const glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

    auto right = glm::cross(front, up);
    if (right.length() < 1e-3) {
        right = eval_front(yaw + 90, 0);
    }

    return glm::normalize(right);
}

auto update_camera(float dt) -> void {
    g_smoothedVelocity = lerp(g_smoothedVelocity, g_velocity, smooth_factor);

    c_pos += g_smoothedVelocity * dt;

    g_view = glm::lookAt(c_pos, c_pos + c_front, c_up_p);
    g_proj = glm::perspective(glm::radians(c_fov), (float) WINDOW_WIDTH / (float) WINDOW_HEIGHT,
                              c_near, c_far);
    g_proj[1][1] *= -1;
}

auto eval_up(glm::vec3 front, glm::vec3 right) -> glm::vec3 {
    return glm::normalize(glm::cross(right, front));
}

glm::vec3 lerp(const glm::vec3 &a, const glm::vec3 &b, float f) {
    return glm::vec3(std::lerp(a.x, b.x, f), std::lerp(a.y, b.y, f), std::lerp(a.z, b.z, f));
}

void cursorPosCallback(GLFWwindow *window, double xpos, double ypos) {
    if (isGuiHovered()) {
        return;
    }

    double dx = xpos - lastX;
    double dy = ypos - lastY;

    lastX = xpos;
    lastY = ypos;

    dy *= c_invert_y ? -1.0f : 1.0f;

    // only work when left button is pressed
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS) {
        return;
    }

    // angle lock
    if (pitch + dy * sensitivity > 89.0f) {
        pitch = 89.0f;
    } else if (pitch + dy * sensitivity < -89.0f) {
        pitch = -89.0f;
    } else {
        pitch += dy * sensitivity;
    }

    if (yaw + dx * sensitivity > 360.0f) {
        yaw = yaw + dx * sensitivity - 360.0f;
    } else if (yaw + dx * sensitivity < 0.0f) {
        yaw = yaw + dx * sensitivity + 360.0f;
    } else {
        yaw += dx * sensitivity;
    }

    c_front = eval_front(yaw, pitch);
    c_right = eval_right(c_front, yaw, pitch);
    c_up_p  = eval_up(c_front, c_right);
}

void scrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
    if (isGuiHovered()) {
        return;
    }

    c_fov -= yoffset * 0.1f;
    if (c_fov < 1.0f) {
        c_fov = 1.0f;
    } else if (c_fov > 90.0f) {
        c_fov = 90.0f;
    }
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    g_velocity = glm::vec3(0.0f, 0.0f, 0.0f);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        g_velocity += speed * c_front;
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        g_velocity -= speed * c_front;
    }

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        g_velocity -= speed * c_right;
    }

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        g_velocity += speed * c_right;
    }

    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        g_velocity += speed * c_up_p;
    }

    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        g_velocity -= speed * c_up_p;
    }

    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        g_velocity *= speed_up;
    }
}

// if gui hovered, mouse input should be ignored
bool isGuiHovered() noexcept {
    ImGuiIO &io = ImGui::GetIO();
    return io.WantCaptureMouse;
}

void light_gui() noexcept {
    ImGui::PushID("Light");
    if (ImGui::CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Position", &g_light_pos.x, -10.0f, 10.0f);
        ImGui::SliderFloat3("Direction", &g_light_dir.x, -1.0f, 1.0f);
        ImGui::SliderFloat3("Color", &g_light_col.x, 0.0f, 1.0f);
    }
    ImGui::PopID();
}

void camera_gui() noexcept {
    ImGui::PushID("Camera");
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Position", &c_pos[0], -10.0f, 10.0f);
        ImGui::SliderFloat3("Front", &c_front[0], -1.0f, 1.0f);
        ImGui::SliderFloat3("Up", &c_up_p[0], -1.0f, 1.0f);
        ImGui::SliderFloat3("Right", &c_right[0], -1.0f, 1.0f);

        ImGui::Separator();

        ImGui::SliderFloat("Yaw", &yaw, 0.0f, 360.0f);
        ImGui::SliderFloat("Pitch", &pitch, -89.0f, 89.0f);
        ImGui::SliderFloat("FOV", &c_fov, 1.0f, 90.0f);
        ImGui::SliderFloat("Near", &c_near, 0.0f, 0.1f);
        ImGui::SliderFloat("Far", &c_far, 0.0f, 100.0f);
        ImGui::SliderFloat("Speed", &speed, 0.0f, 10.0f);
        ImGui::SliderFloat("Speed Up", &speed_up, 1.0f, 10.0f);
        ImGui::SliderFloat("Sensitivity", &sensitivity, 0.0f, 1.0f);
        ImGui::Checkbox("Invert Y", &c_invert_y);
    }
    ImGui::PopID();
}
