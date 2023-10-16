#pragma once

#include <cstdint>
#include <limits>
#include <stdint.h>
#include <vector>

struct Bounds {
    float min[3]{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max()};
    float max[3]{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                 std::numeric_limits<float>::lowest()};
};

struct SphereBounds {
    float center[3]{0.0f, 0.0f, 0.0f};
    float radius{0.0f};
};

struct Cone {
    float apex[3];
    float axis[3];
    float cutoff; // cos(angle/2)
};

struct BuilderSettings {
    bool  enable             = false;
    bool  preserve_area      = false;
    bool  explicit_tangents  = false;
    float preserve_triangles = 1.0F;
};

/**
 * @brief VertexConfigure
 * 1. position float x3
 * 2. other attributes
 */
struct VertexConfigure {
    std::vector<uint32_t> attr_strides;
    std::vector<float>    attr_weights;
};

/**
 * @brief Cluster data struct (Header)
 * problems:
 * 1. for sub-mesh with different material but the same boundary, the sub-mesh boundary might crack,
 */
struct Cluster {
    // cluster properties
    float    lodError{0.0f};
    uint32_t material_id{0u};
    uint32_t vertex_config{0u};
    uint32_t vertex_count{0u};
    uint32_t index_count{0u};

    Bounds       bounds;
    SphereBounds sphere_bounds;

    // hierarchy data
    uint32_t group_id{0u};
    uint32_t generating_group_id{0u};
    uint32_t lodLevel{0u};

    // float edge_length;
    // float surface_area;

    union {
        struct { // pointer to cluster data
            float *   vertices;
            uint32_t *indices;
        };
        struct { // recording streaming offset for each cluster
            uint32_t vert_offset;
            uint32_t index_offset;
        };
    } data;
    uint32_t mask{0u}; // can determine whether cluster is loaded to memory
};

/**
 * @brief cluster group
 *
 */
struct ClusterGroup {
    Bounds       bounds;
    SphereBounds sphere_bounds;

    float    maxParentError{0.f};
    float    minLodError{0.f};
    uint32_t lodLevel{0u};

    std::vector<uint32_t> children; // cluster id
};

/**
 * @brief Data stream that contains whole Meshbundle, useful for data streaming
 *
 */
struct DataStream {
    // header
    std::vector<VertexConfigure> vertex_configures;
    std::vector<Cluster>         clusters;
    std::vector<ClusterGroup>    cluster_groups;

    // data stream
    FILE *               fstream{nullptr};
    std::vector<uint8_t> buffer;
};

void build(float *vertices, uint32_t num_vertices, std::vector<uint32_t> indices, bool has_color,
           bool has_tangent, uint32_t uv_count, BuilderSettings settings, DataStream &stream);