#include "builder.hpp"

#include <boost/program_options.hpp>

#include <filesystem>
#include <iostream>
#include <string>

void loadMesh(std::string path, std::vector<float> &verts, std::vector<uint32_t> &indices,
              bool &has_color, bool &has_tangent, uint32_t &uv_count);

int main(int argc, char **argv) {
    boost::program_options::options_description desc("Allowed options");

    desc.add_options()("help,h", "produce help message") // default help message
        ("input,i", boost::program_options::value<std::string>(),
         "input file") // input file
        ("output,o", boost::program_options::value<std::string>(),
         "output file") // output file
        ("enable,e", boost::program_options::value<bool>(),
         "enable builder") // enable builder
        ("preserve_area", boost::program_options::value<bool>(),
         "preserve area") // preserve area
        ("explicit_tangents", boost::program_options::value<bool>(),
         "explicit tangents") // explicit tangents
        ("preserve_triangles", boost::program_options::value<float>(), "preserve triangles");

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
        std::cout << "input file not specified\n";
        return 1;
    }

    std::string output_file;

    if (vm.count("output")) {
        output_file = vm["output"].as<std::string>();
    } else {
        std::cout << "output file not specified\n";
        return 1;
    }

    BuilderSettings settings{
        .enable             = true,
        .preserve_area      = false,
        .explicit_tangents  = false,
        .preserve_triangles = 1.0F,
    };

    if (vm.count("enable")) {
        settings.enable = vm["enable"].as<bool>();
    }

    if (vm.count("preserve_area")) {
        settings.preserve_area = vm["preserve_area"].as<bool>();
    }

    if (vm.count("explicit_tangents")) {
        settings.explicit_tangents = vm["explicit_tangents"].as<bool>();
    }

    if (vm.count("preserve_triangles")) {
        settings.preserve_triangles = vm["preserve_triangles"].as<float>();
    }

    std::vector<float>    verts;
    std::vector<uint32_t> indices;
    bool                  has_color;
    bool                  has_tangent;
    uint32_t              uv_count;

    if (!std::filesystem::is_regular_file(input_file)) {
        std::cout << "input file \"" << input_file << "\" is not a regular file\n";
        return 1;
    }

    if (std::filesystem::is_regular_file(output_file)) {
        std::filesystem::remove(output_file);
    }

    loadMesh(input_file, verts, indices, has_color, has_tangent, uv_count);

    std::cout << "       file: " << input_file << "\n";
    std::cout << "      verts: " << verts.size() << "\n";
    std::cout << "    indices: " << indices.size() << "\n";
    std::cout << "  has_color: " << has_color << "\n";
    std::cout << "has_tangent: " << has_tangent << "\n";

    DataStream result;

    build(verts.data(), verts.size(), indices, has_color, has_tangent, uv_count, settings, result);

    std::cout << "result size: " << result.buffer.size() << "\n";

    if (result.buffer.size() > 0) {
        FILE *fp = fopen(output_file.c_str(), "wb");
        fwrite(result.buffer.data(), 1, result.buffer.size(), fp);
        fclose(fp);
    } else {
        std::cout << "no data generated\n";
    }

    return 0;
}

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

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

    auto vertex_count = attrib.vertices.size() / 3;
    bool has_texcoord = attrib.texcoords.size() > 0;
    has_color         = attrib.colors.size() > 0;
    has_tangent       = false;
    uv_count          = has_texcoord ? 1 : 0;

    uint32_t vertex_size = 3 + 3 + (has_tangent ? 4 : 0) + (has_color ? 4 : 0) + uv_count * 2;

    verts.reserve(vertex_count * vertex_size);
    indices.reserve(vertex_count * 3);

    for (const auto &shape : shapes) {
        for (const auto &index : shape.mesh.indices) {
            verts.insert(verts.end(), attrib.vertices.begin() + 3 * index.vertex_index + 1,
                         attrib.vertices.begin() + 3 * index.vertex_index + 4);

            verts.insert(verts.end(), attrib.normals.begin() + 3 * index.normal_index + 1,
                         attrib.normals.begin() + 3 * index.normal_index + 4);

            if (has_color) {
                verts.insert(verts.end(), attrib.colors.begin() + 3 * index.vertex_index + 1,
                             attrib.colors.begin() + 3 * index.vertex_index + 4);
                verts.push_back(1.0F);
            }

            if (has_texcoord) {
                verts.insert(verts.end(), attrib.texcoords.begin() + 2 * index.texcoord_index,
                             attrib.texcoords.begin() + 2 * index.texcoord_index + 2);
            }

            indices.push_back(index.vertex_index);
        }
    }
}
