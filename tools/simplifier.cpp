
#include "simplifier.hpp"
#include "builder.hpp"
#include "config.h"

#include <boost/program_options.hpp>

#include <filesystem>
#include <iostream>
#include <string>

#define MESH_PATH FILESYSTEM_ROOT "/visualizer/models/dragon.obj"

void loadMesh(std::string path, std::vector<float> &verts, std::vector<uint32_t> &indices,
              bool &has_color, bool &has_tangent, uint32_t &uv_count);

void dumpMesh(std::string_view path, MeshDataView &view);

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

    std::string input_file;

    if (vm.count("input")) {
        std::filesystem::path path = vm["input"].as<std::string>();
        input_file                 = path.string();
    } else {
        std::cout << "input file not specified, use default\n";
        input_file = MESH_PATH;
    }

    if (input_file.find(".obj") == std::string::npos) {
        std::cout << "input file must be .obj\n";
        return 1;
    }

    if (!std::filesystem::exists(input_file)) {
        std::cout << "input file does not exist\n";
        return 1;
    }

    std::vector<float>    verts;
    std::vector<uint32_t> indices;
    bool                  has_color;
    bool                  has_tangent;
    uint32_t              uv_count;

    loadMesh(input_file, verts, indices, has_color, has_tangent, uv_count);

    std::cout << "       file: " << input_file << "\n";
    std::cout << "      verts: " << verts.size() << "\n";
    std::cout << "    indices: " << indices.size() << "\n";
    std::cout << "  has_color: " << has_color << "\n";
    std::cout << "has_tangent: " << has_tangent << "\n";

    std::vector<uint32_t> attr_strides{3};    // normal
    std::vector<float>    attr_weights{1.0f}; // normal

    if (has_color) {
        attr_strides.push_back(4);
        attr_weights.push_back(1.0);
    }
    if (has_tangent) {
        attr_strides.push_back(4);
        attr_weights.push_back(1.0);
    }

    const auto vertex_stride = std::accumulate(attr_strides.begin(), attr_strides.end(), 3U);

    const auto num_vertices = verts.size() / vertex_stride;

    MeshDataView view(std::span(verts), num_vertices, vertex_stride, std::span(indices),
                      attr_strides, attr_weights);

    MeshSimplify simplifier(view, false);

    // for (auto i = 0U; i < num_vertices; ++i) {
    //     simplifier.lock_position(i);
    // }

    simplifier.simplify(80000, 1.0f);

    simplifier.dump_obj("out.obj");

    return 0;
}

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

struct Vertex {
    Vec3 pos;
    Vec3 normal;

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

void loadMesh(std::string path, std::vector<float> &verts, std::vector<uint32_t> &indices,
              bool &has_color, bool &has_tangent, uint32_t &uv_count) {
    // read obj using tinyobjloader
    tinyobj::attrib_t                attrib;
    std::vector<tinyobj::shape_t>    shapes;
    std::vector<tinyobj::material_t> materials;
    std::string                      warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.data())) {
        throw std::runtime_error(warn + err);
    }

    std::unordered_map<Vertex, uint32_t, VertexHasher> uniqueVertices{};

    auto vertex_count = attrib.vertices.size() / 3;
    auto index_count
        = std::accumulate(shapes.begin(), shapes.end(), 0ULL, [](auto acc, const auto &shape) {
              return acc + shape.mesh.indices.size();
          });

    has_color   = false;
    has_tangent = false;
    uv_count    = 0;

    uint32_t vertex_size = 3 + 3;

    std::vector<Vertex> vertices;
    vertices.reserve(vertex_count);
    indices.reserve(vertex_count * 3);

    for (const auto &shape : shapes) {
        for (const auto &index : shape.mesh.indices) {
            Vertex vertex{};
            vertex.pos = {attrib.vertices[3 * index.vertex_index + 0],
                          attrib.vertices[3 * index.vertex_index + 1],
                          attrib.vertices[3 * index.vertex_index + 2]};

            vertex.normal = {attrib.normals[3 * index.normal_index + 0],
                             attrib.normals[3 * index.normal_index + 1],
                             attrib.normals[3 * index.normal_index + 2]};

            if (uniqueVertices.count(vertex) == 0) {
                uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                vertices.push_back(vertex);
            }

            indices.push_back(uniqueVertices[vertex]);
        }
    }

    verts.resize(vertices.size() * vertex_size);

    for (size_t i = 0; i < vertices.size(); ++i) {
        const auto &vertex         = vertices[i];
        verts[i * vertex_size + 0] = vertex.pos.x;
        verts[i * vertex_size + 1] = vertex.pos.y;
        verts[i * vertex_size + 2] = vertex.pos.z;
        verts[i * vertex_size + 3] = vertex.normal.x;
        verts[i * vertex_size + 4] = vertex.normal.y;
        verts[i * vertex_size + 5] = vertex.normal.z;
    }
}

#include <ranges>

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