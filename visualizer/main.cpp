#include "../src/builder.hpp"
#include "../src/cluster.hpp"
#include "../src/cluster_group.hpp"
#include "../src/hasher.hpp"
#include "../src/partitioner.hpp"
#include "../src/simplify.hpp"
#include "../src/utility.hpp"

#include <bit>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/sequential_vertex_coloring.hpp>
#include <boost/pending/property.hpp>

#include <glm/ext/quaternion_geometric.hpp>
#include <imgui.h>
#include <mutex>
#include <numeric>
#include <string>
#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <boost/program_options.hpp>

#include "config.h"
#include "render.h"
#include "viewer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <coroutine>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <ranges>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#define MESH_PATH FILESYSTEM_ROOT "/visualizer/models/dragon.obj"

void dumpMesh(std::string_view path, MeshDataView &view) {
    std::ofstream file(path.data());

    if (!file.is_open()) {
        std::cout << "failed to open file: " << path << "\n";
        return;
    }

    auto &indices = view.indices;

    for (auto i : std::views::iota(0U, view.vertex_count)) {
        const float *position = view.get_vertex(i);
        file << "v " << position[0] << " " << position[1] << " " << position[2] << "\n";
    }

    for (size_t i = 0; i < indices.size(); i += 3) {
        file << "f " << indices[i + 0] + 1 << " " << indices[i + 1] + 1 << " " << indices[i + 2] + 1
             << "\n";
    }

    file.close();
}

void createPlane(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices, uint32_t res,
                 float size = 10.0f) {
    // generate triangle vertices for a flat plane
    res = std::max(res, 2U);
    vertices.clear();
    indices.clear();
    vertices.reserve(res * res);
    indices.reserve((res - 1) * (res - 1) * 6);

    for (auto z = 0U; z < res; ++z) {
        for (auto x = 0U; x < res; ++x) {
            Vertex vertex{};

            vertex.pos = {size * (x / static_cast<float>(res - 1) - 0.5f), 0.0f,
                          size * (z / static_cast<float>(res - 1) - 0.5f)};

            vertex.normal = {0.0f, 1.0f, 0.0f};
            vertices.push_back(vertex);
        }
    }

    // generate indices
    for (auto z = 0U; z < res - 1; ++z) {
        for (auto x = 0U; x < res - 1; ++x) {
            auto a = x + z * res;
            auto b = (x + 1) + (z * res);
            auto c = x + ((z + 1) * res);
            auto d = (x + 1) + ((z + 1) * res);

            // triangle indices
            indices.push_back(c);
            indices.push_back(b);
            indices.push_back(a);

            indices.push_back(c);
            indices.push_back(d);
            indices.push_back(b);
        }
    }
}

void createSphere(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices, uint32_t res,
                  float size = 10.0f) {
    // generate sphere
    res = std::max(res, 3U);

    constexpr float pi_cached = std::numbers::pi_v<float>;

    vertices.clear();
    indices.clear();
    vertices.reserve(res * res);
    indices.reserve((res - 1) * (res - 1) * 6);

    for (auto y = 0U; y < res; ++y) {
        for (auto x = 0U; x < res; ++x) {
            Vertex vertex{};

            auto x_ = std::cos(2 * pi_cached * x / (res - 1)) * std::sin(pi_cached * y / (res - 1));
            auto y_ = std::cos(pi_cached * y / (res - 1));
            auto z_ = std::sin(2 * pi_cached * x / (res - 1)) * std::sin(pi_cached * y / (res - 1));

            vertex.pos = {size * x_, size * y_, size * z_};

            vertex.normal = {x_, y_, z_};
            vertices.push_back(vertex);
        }
    }

    // generate indices
    for (auto y = 0U; y < res - 1; ++y) {
        for (auto x = 0U; x < res - 1; ++x) {
            auto a = x + y * res;
            auto b = (x + 1) + (y * res);
            auto c = x + ((y + 1) * res);
            auto d = (x + 1) + ((y + 1) * res);

            // triangle indices
            indices.push_back(a);
            indices.push_back(b);
            indices.push_back(c);

            indices.push_back(b);
            indices.push_back(d);
            indices.push_back(c);
        }
    }
}

void createBoxWire(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices, glm::vec3 min,
                   glm::vec3 max) {
    vertices.clear();
    vertices.reserve(8);

    vertices.push_back({min.x, min.y, min.z});
    vertices.push_back({min.x, min.y, max.z});
    vertices.push_back({min.x, max.y, min.z});
    vertices.push_back({min.x, max.y, max.z});
    vertices.push_back({max.x, min.y, min.z});
    vertices.push_back({max.x, min.y, max.z});
    vertices.push_back({max.x, max.y, min.z});
    vertices.push_back({max.x, max.y, max.z});

    indices = {
        0, 1, 1, 3, 3, 2, 2, 0, // bottom
        4, 5, 5, 7, 7, 6, 6, 4, // top
        0, 4, 1, 5, 2, 6, 3, 7, // sides
    };
}

void loadMesh(const std::string_view mesh_path, std::vector<Vertex> &vertices,
              std::vector<uint32_t> &indices) {
    tinyobj::attrib_t                attrib;
    std::vector<tinyobj::shape_t>    shapes;
    std::vector<tinyobj::material_t> materials;
    std::string                      warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, mesh_path.data())) {
        throw std::runtime_error(warn + err);
    }

    std::unordered_map<Vertex, uint32_t, VertexHasher> uniqueVertices{};

    Vec3 min = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max()};
    Vec3 max = {std::numeric_limits<float>::min(), std::numeric_limits<float>::min(),
                std::numeric_limits<float>::min()};

    // reserve space
    vertices.reserve(attrib.vertices.size() / 3);
    auto index_count
        = std::accumulate(shapes.begin(), shapes.end(), 0ULL, [](auto acc, const auto &shape) {
              return acc + shape.mesh.indices.size();
          });
    indices.reserve(index_count);

    for (const auto &shape : shapes) {
        // bounding box

        for (const auto &index : shape.mesh.indices) {
            Vertex vertex{};

            vertex.pos = {attrib.vertices[3 * index.vertex_index + 0],
                          attrib.vertices[3 * index.vertex_index + 1],
                          attrib.vertices[3 * index.vertex_index + 2]};

            vertex.normal = {attrib.normals[3 * index.normal_index + 0],
                             attrib.normals[3 * index.normal_index + 1],
                             attrib.normals[3 * index.normal_index + 2]};

            min.x = std::min(min.x, vertex.pos.x);
            min.y = std::min(min.y, vertex.pos.y);
            min.z = std::min(min.z, vertex.pos.z);

            max.x = std::max(max.x, vertex.pos.x);
            max.y = std::max(max.y, vertex.pos.y);
            max.z = std::max(max.z, vertex.pos.z);

            if (uniqueVertices.count(vertex) == 0) {
                uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                vertices.push_back(vertex);
            }

            indices.push_back(uniqueVertices[vertex]);
        }
    }

    auto max_extent = std::max({max.x - min.x, max.y - min.y, max.z - min.z});

    // normalize
    for (auto &vertex : vertices) {
        vertex.pos.y -= min.y;
        vertex.pos = vertex.pos / max_extent;
        vertex.pos.y += 0.01f;
    }
}

template <typename T, typename PositionGetter>
struct PositionHashMapAdapter {
    PositionHashMapAdapter(PositionGetter getter, uint32_t size = 0) :
        map(size, HashAdapter(getter), EqualAdapter(getter)) {
    }

    struct HashAdapter {
        PositionGetter getter;

        HashAdapter(PositionGetter getter) : getter(getter){};
        HashAdapter(const HashAdapter &) = default;
        HashAdapter(HashAdapter &&)      = default;
        HashAdapter &operator=(const HashAdapter &) = default;
        HashAdapter &operator=(HashAdapter &&) = default;
        ~HashAdapter()                         = default;

        size_t operator()(uint32_t i) const {
            return PositionHasher()(getter(i));
        }
    };

    struct EqualAdapter {
        PositionGetter getter;

        EqualAdapter(PositionGetter getter) : getter(getter){};
        EqualAdapter(const EqualAdapter &) = default;
        EqualAdapter(EqualAdapter &&)      = default;
        EqualAdapter &operator=(const EqualAdapter &) = default;
        EqualAdapter &operator=(EqualAdapter &&) = default;
        ~EqualAdapter()                          = default;

        bool operator()(const uint32_t l, const uint32_t r) const {
            return getter(l) == getter(r);
        }
    };

    std::unordered_multimap<uint32_t, T, HashAdapter, EqualAdapter> map;
};

void loadScene(Scene &scene, std::string_view mesh_path, Render &render) {
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;

    static_assert(sizeof(glm::vec3) == 3 * sizeof(float));

    loadMesh(mesh_path, vertices, indices);

    std::vector<uint32_t> attr_stride{3};     // normal
    std::vector<float>    attr_weights{1.0f}; // normal
    std::span<float>      vert = std::span<float>(reinterpret_cast<float *>(vertices.data()),
                                             vertices.size() * sizeof(Vertex));

    MeshDataView view(vert, vertices.size(), 6, indices, attr_stride, attr_weights);

    dumpMesh("out.obj", view);

    auto clusters = std::vector<ClusterImpl>{};

    auto mesh_bound   = Bounds{};
    auto sphere_bound = SphereBounds{};

    cluster_triangles(view, clusters, mesh_bound, sphere_bound);
    std::cout << "Cluster Count: " << clusters.size() << std::endl;

    auto start = scene.meshes.size();

    std::unordered_multimap<Edge, uint32_t, EdgeHasher> external_edge_map;

    auto get_position = [&](const ClusterImpl &cluster, uint32_t index) -> const Position * {
        return reinterpret_cast<const Position *>(
            cluster.view.get_vertex(cluster.index_array[index]));
    };

    auto get_vertex = [&](const ClusterImpl &cluster, uint32_t index) -> const Vertex * {
        return reinterpret_cast<const Vertex *>(
            cluster.view.get_vertex(cluster.index_array[index]));
    };

    vertices.clear();
    indices.clear();

    for (uint32_t i = 0; i < clusters.size(); i++) {
        auto &cluster = clusters[i];
        auto  external_edges
            = std::views::iota(0U, cluster.index_array.size())
              | std::views::filter([&](auto edge_idx) { return cluster.is_external[edge_idx]; });

        for (auto e : external_edges) {
            external_edge_map.emplace(
                Edge{*get_position(cluster, e), *get_position(cluster, triangle_cycle(e))}, i);
            vertices.push_back(*get_vertex(cluster, e));
            vertices.push_back(*get_vertex(cluster, triangle_cycle(e)));
        }
    }

    std::cout << "External Edge Count: " << external_edge_map.size() << std::endl;

    // build a cluster adj graph
    using ClusterAdj
        = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property,
                                boost::no_property, boost::disallow_parallel_edge_tag>;

    auto cluster_adj = ClusterAdj(clusters.size());

    for (auto [edge, cluster_idx] : external_edge_map) {
        auto adj_edge     = Edge{edge.p1, edge.p0};
        auto [begin, end] = external_edge_map.equal_range(adj_edge);
        for (auto it = begin; it != end; ++it) {
            auto other_cluster_idx = it->second;
            if (cluster_idx != other_cluster_idx) {
                auto [e, res] = boost::add_edge(cluster_idx, other_cluster_idx, cluster_adj);
                if (!res) {
                    std::cout << "Edge already exists" << std::endl;
                }
            }
        }
    }

    auto color_result = std::vector<uint32_t>(clusters.size());
    auto color_map    = boost::make_safe_iterator_property_map(
        color_result.begin(), color_result.size(), boost::get(boost::vertex_index, cluster_adj));
    auto color_count = boost::sequential_vertex_coloring(cluster_adj, color_map);

    std::cout << "Color Count: " << color_count << std::endl;

    indices.resize(vertices.size());
    std::iota(indices.begin(), indices.end(), 0);
    emplace_mesh(scene, render, "External Edge", vertices, indices, TopologyType::LINE,
                 RasterizeType::WIREFRAME);

    scene.meshes.reserve(clusters.size());
    scene.clusters.reserve(clusters.size());
    for (uint32_t i = 0; i < clusters.size(); i++) {
        auto &cluster     = clusters[i];
        auto  vertex_span = std::span(reinterpret_cast<Vertex *>(cluster.vertex_array.data()),
                                     cluster.vertex_array.size() / cluster.view.vertex_stride);
        auto  index_span  = cluster.index_array;

        emplace_mesh(scene, render, ("Cluster " + std::to_string(i)), vertex_span, index_span,
                     TopologyType::TRIANGLE, RasterizeType::FILLED);

        auto &instance              = scene.meshes.back();
        instance.pso.clusterColorId = color_result[i];
        instance.pso.model          = glm::mat4(1.0f);
    }

    scene.flags.resize(scene.meshes.size(), Scene::Mask::enabled);
}

void emplace_mesh(Scene &scene, Render &render, std::string name, std::span<Vertex> vertices,
                  std::span<uint32_t> indices, TopologyType topology, RasterizeType raster) {
    auto &instance = scene.meshes.emplace_back(name, render, vertices, indices, topology, raster);
    scene.flags.emplace_back(Scene::Mask::enabled);
    auto &clusterData = scene.clusters.emplace_back();
    clusterData.indices.resize(indices.size());
    clusterData.vertices.resize(vertices.size());
    std::copy(indices.begin(), indices.end(), clusterData.indices.begin());
    std::transform(vertices.begin(), vertices.end(), clusterData.vertices.begin(),
                   [](auto &vertex) { return vertex.pos; });
    for (int i = 0; i < 3; i++) {
        instance.bound.min[i] = std::numeric_limits<float>::max();
        instance.bound.max[i] = std::numeric_limits<float>::lowest();
    }
    for (auto &vertex : clusterData.vertices) {
        for (auto i = 0; i < 3; i++) {
            instance.bound.min[i] = std::min(instance.bound.min[i], vertex.data[i]);
            instance.bound.max[i] = std::max(instance.bound.max[i], vertex.data[i]);
        }
    }
    auto box_v = std::vector<Vertex>{};
    auto box_i = std::vector<uint32_t>{};

    createBoxWire(box_v, box_i,
                  glm::vec3(instance.bound.min[0], instance.bound.min[1], instance.bound.min[2]),
                  glm::vec3(instance.bound.max[0], instance.bound.max[1], instance.bound.max[2]));
    auto &bound     = scene.boundingBoxes.emplace_back("BoundingBox", render, box_v, box_i,
                                                   TopologyType::LINE, RasterizeType::WIREFRAME);
    bound.lineWidth = 2.0f;
}

void renderScene(Scene &scene, VkCommandBuffer commandBuffer, uint32_t index = 0) {
    auto enabled_meshes
        = std::views::iota(0U, scene.meshes.size())
          | std::views::filter([&](auto i) { return scene.flags[i] & Scene::Mask::enabled; })
          | std::views::transform([&](auto i) { return &scene.meshes[i]; });
    for (const auto &mesh : enabled_meshes) {
        mesh->render(index);
    }
    auto enabled_bounds
        = std::views::iota(0U, scene.boundingBoxes.size())
          | std::views::filter([&](auto i) { return scene.flags[i] & Scene::Mask::bounded; })
          | std::views::transform([&](auto i) { return &scene.boundingBoxes[i]; });
    for (const auto &mesh : enabled_bounds) {
        mesh->render(index);
    }
}

#include <hasher.hpp>

int main(int argc, char **argv) {
    boost::program_options::options_description desc("Allowed options");

    desc.add_options()("help,h", "produce help message") // default help message
        ("input,i", boost::program_options::value<std::string>(), "input file");

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    auto mesh_path = std::string(MESH_PATH);

    if (vm.count("input")) {
        mesh_path = vm["input"].as<std::string>();
    } else {
        std::cout << "Input file was not set. use default\n";
    }

    auto render = Render("Visualizer");

    Scene scene;
    Timer timer("load scene");
    {
        std::vector<Vertex>   vertices;
        std::vector<uint32_t> indices;
        createPlane(vertices, indices, 10, 5.0f);
        emplace_mesh(scene, render, "Ground", vertices, indices, TopologyType::TRIANGLE,
                     RasterizeType::FILLED);
        scene.meshes.back().pso.model = glm::mat4(1.0f);
        createSphere(vertices, indices, 50, 0.05f);
        emplace_mesh(scene, render, "Sphere", vertices, indices, TopologyType::TRIANGLE,
                     RasterizeType::FILLED);
        scene.meshes.back().pso.model = glm::translate(glm::mat4(1.0f), {0.0f, 100.0f, 0.0f});
        auto &bound                   = scene.meshes.back().bound;
        for (int i = 0; i < 3; i++) {
            bound.min[i] = std::numeric_limits<float>::max();
            bound.max[i] = std::numeric_limits<float>::lowest();
        }
        // createBoxWire(vertices, indices, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
        // emplace_mesh(scene, render, "Box", vertices, indices, TopologyType::LINE,
        //              RasterizeType::WIREFRAME);
    }
    loadScene(scene, mesh_path, render);

    auto recordGui = [&] {
        newImGuiFrame();

        ImGui::ShowStackToolWindow();

        // fps counter using overlay
        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::SetNextWindowSize(ImVec2(300, WINDOW_HEIGHT - 20));

        ImGui::Begin("Inspector", nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
                         | ImGuiWindowFlags_NoFocusOnAppearing);

        ImGui::Text("FPS: %.1f (%.2f ms)", render.fps, render.ms);
        ImGui::Text("Frame Count: %llu", render.frameCount);

        ImGui::Spacing();
        ImGui::Separator();

        static char                     input[256] = {MESH_PATH};
        static std::vector<std::string> histPool;
        static auto                     it = histPool.begin();

        camera_gui();

        light_gui();

        scene_gui(scene);

        ImGui::End();

        endImGuiFrame();
    };

    auto res = render.loop(
        [&] {
            const auto &dt = render.dt;
            update_camera(dt);

            recordGui();
        },
        [&] {
            auto        currentBuffer = render.currentBuffer;
            const auto &commandBuffer = render.commandBuffers[currentBuffer];
            renderScene(scene, commandBuffer, currentBuffer);
            renderImGui(commandBuffer); // imgui at the end
        },
        [] {});

    return res;
}