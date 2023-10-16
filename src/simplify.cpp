#include "simplify.hpp"

#include "meshopt_simplify.cpp"
#include "meshoptimizer.h"
#include "simplify.hpp"
#include "utility.hpp"

#include <fstream>
#include <iostream>
#include <ranges>

meshopt_Allocator MeshSimplify::m_allocator;

MeshSimplify::MeshSimplify(MeshDataView view, bool inplace) : input_view(view), m_inplace(inplace) {
    if (inplace) {
        this->view = view;
    } else {
        m_indices = std::vector<uint32_t>(view.indices.begin(), view.indices.end());
        // m_vertices          = std::vector<float>(view.vertices.size());
        this->view = view;
        // this->view.vertices = std::span(m_vertices);
        this->view.indices = std::span(m_indices);
    }

    assert(view.indices.size() % 3 == 0);
}

void MeshSimplify::lock_position(uint32_t i) {
    locked_positions.emplace_back(i);
}

float MeshSimplify::simplify(size_t target_num_tri, float target_error) {
    EdgeAdjacency adjacency = {};
    meshopt::prepareEdgeAdjacency(adjacency, input_view.indices.size(), input_view.vertex_count,
                                  m_allocator);
    meshopt::updateEdgeAdjacency(adjacency, input_view.indices.data(), input_view.indices.size(),
                                 input_view.vertex_count, NULL);

    size_t vertex_count             = input_view.vertex_count;
    size_t vertex_positions_stride  = input_view.vertex_stride * sizeof(float);
    size_t vertex_attributes_stride = (input_view.vertex_stride - 3) * sizeof(float);
    size_t index_count              = input_view.indices.size();

    auto attribute_weight = std::vector<float>();
    attribute_weight.reserve(input_view.vertex_stride - 3);
    for (auto i : std::views::iota(0U, input_view.attr_stride.size())) {
        auto stride = input_view.attr_stride[i];
        attribute_weight.insert(attribute_weight.end(), stride, input_view.attr_weight[i]);
    }

    assert(attribute_weight.size() == input_view.vertex_stride - 3);

    float *vertex_positions_data  = input_view.get_vertex(0);
    float *vertex_attributes_data = input_view.get_attr(0);
    float *attribute_weights      = attribute_weight.data();

    unsigned int *remap = m_allocator.allocate<unsigned int>(vertex_count);
    unsigned int *wedge = m_allocator.allocate<unsigned int>(vertex_count);
    meshopt::buildPositionRemap(remap, wedge, vertex_positions_data, vertex_count,
                                vertex_positions_stride, m_allocator);

    unsigned char *vertex_kind = m_allocator.allocate<unsigned char>(vertex_count);
    unsigned int * loop        = m_allocator.allocate<unsigned int>(vertex_count);
    unsigned int * loopback    = m_allocator.allocate<unsigned int>(vertex_count);
    meshopt::classifyVertices(vertex_kind, loop, loopback, vertex_count, adjacency, remap, wedge,
                              0);

#if TRACE
    size_t unique_positions = 0;
    for (size_t i = 0; i < vertex_count; ++i)
        unique_positions += remap[i] == i;

    printf("position remap: %d vertices => %d positions\n", int(vertex_count),
           int(unique_positions));

    size_t kinds[meshopt::Kind_Count] = {};
    for (size_t i = 0; i < vertex_count; ++i)
        kinds[vertex_kind[i]] += remap[i] == i;

    printf("kinds: manifold %d, border %d, seam %d, complex %d, locked %d\n",
           int(kinds[meshopt::Kind_Manifold]), int(kinds[meshopt::Kind_Border]),
           int(kinds[meshopt::Kind_Seam]), int(kinds[meshopt::Kind_Complex]),
           int(kinds[meshopt::Kind_Locked]));
#endif

    lock_positions();
    for (auto i : locked_positions) {
        vertex_kind[i] = meshopt::Kind_Locked;
    }

    meshopt::Vector3 *vertex_positions = m_allocator.allocate<meshopt::Vector3>(vertex_count);
    meshopt::rescalePositions(vertex_positions, vertex_positions_data, vertex_count,
                              vertex_positions_stride);

    float *vertex_attributes = NULL;
    size_t attribute_count
        = std::accumulate(input_view.attr_stride.begin(), input_view.attr_stride.end(), 0U);

    // vertex_attributes_data = nullptr;
    // attribute_weights      = nullptr;
    // attribute_count        = 0;

    if (attribute_count) {
        vertex_attributes = m_allocator.allocate<float>(vertex_count * attribute_count);
        meshopt::rescaleAttributes(vertex_attributes, vertex_attributes_data, vertex_count,
                                   vertex_attributes_stride, attribute_weights, attribute_count);
    }

    meshopt::Quadric *vertex_quadrics = m_allocator.allocate<meshopt::Quadric>(vertex_count);
    memset(vertex_quadrics, 0, vertex_count * sizeof(meshopt::Quadric));

    meshopt::Quadric *    attribute_quadrics  = NULL;
    meshopt::QuadricGrad *attribute_gradients = NULL;

    if (attribute_count) {
        attribute_quadrics = m_allocator.allocate<meshopt::Quadric>(vertex_count);
        memset(attribute_quadrics, 0, vertex_count * sizeof(meshopt::Quadric));

        attribute_gradients
            = m_allocator.allocate<meshopt::QuadricGrad>(vertex_count * attribute_count);
        memset(attribute_gradients, 0,
               vertex_count * attribute_count * sizeof(meshopt::QuadricGrad));
    }

    meshopt::fillFaceQuadrics(vertex_quadrics, input_view.indices.data(), index_count,
                              vertex_positions, remap);
    meshopt::fillEdgeQuadrics(vertex_quadrics, input_view.indices.data(), index_count,
                              vertex_positions, remap, vertex_kind, loop, loopback);

    if (attribute_count)
        meshopt::fillAttributeQuadrics(attribute_quadrics, attribute_gradients,
                                       input_view.indices.data(), index_count, vertex_positions,
                                       vertex_attributes, attribute_count, remap);

    size_t collapse_capacity
        = meshopt::boundEdgeCollapses(adjacency, vertex_count, index_count, vertex_kind);

    meshopt::Collapse *edge_collapses  = m_allocator.allocate<meshopt::Collapse>(collapse_capacity);
    unsigned int *     collapse_order  = m_allocator.allocate<unsigned int>(collapse_capacity);
    unsigned int *     collapse_remap  = m_allocator.allocate<unsigned int>(vertex_count);
    unsigned char *    collapse_locked = m_allocator.allocate<unsigned char>(vertex_count);

    size_t result_count = index_count;
    float  result_error = 0;

    float error_limit = target_error * target_error;

    size_t    target_index_count = target_num_tri * 3;
    uint32_t *result             = view.indices.data();

    while (result_count > target_index_count) {
        // note: throughout the simplification process adjacency structure reflects welded topology
        // for result-in-progress
        meshopt::updateEdgeAdjacency(adjacency, result, result_count, vertex_count, remap);

        size_t edge_collapse_count = pickEdgeCollapses(edge_collapses, collapse_capacity, result,
                                                       result_count, remap, vertex_kind, loop);
        assert(edge_collapse_count <= collapse_capacity);

        // no edges can be collapsed any more due to topology restrictions
        if (edge_collapse_count == 0) {
            std::cout << "Failed to simplify due to topology restrictions\n";
            break;
        }

        rankEdgeCollapses(edge_collapses, edge_collapse_count, vertex_positions, vertex_attributes,
                          vertex_quadrics, attribute_quadrics, attribute_gradients, attribute_count,
                          remap);

        sortEdgeCollapses(collapse_order, edge_collapses, edge_collapse_count);

        size_t triangle_collapse_goal = (result_count - target_index_count) / 3;

        for (size_t i = 0; i < vertex_count; ++i)
            collapse_remap[i] = unsigned(i);

        memset(collapse_locked, 0, vertex_count);

        size_t collapses = performEdgeCollapses(
            collapse_remap, collapse_locked, vertex_quadrics, attribute_quadrics,
            attribute_gradients, attribute_count, edge_collapses, edge_collapse_count,
            collapse_order, remap, wedge, vertex_kind, vertex_positions, adjacency,
            triangle_collapse_goal, error_limit, result_error);

        // no edges can be collapsed any more due to hitting the error limit or triangle collapse
        // limit
        if (collapses == 0) {
            std::cout
                << "Failed to simplify due to hitting the error limit or triangle collapse limit\n";
            break;
        }

        meshopt::remapEdgeLoops(loop, vertex_count, collapse_remap);
        meshopt::remapEdgeLoops(loopback, vertex_count, collapse_remap);

        size_t new_count = meshopt::remapIndexBuffer(result, result_count, collapse_remap);
        assert(new_count < result_count);

        result_count = new_count;
    }

    if (result_count > target_index_count) {
        std::cout << "Failed to simplify to target index count, target is " << target_index_count
                  << " but got " << result_count << std::endl;
    }

    view.indices = std::span(result, result_count);

    return std::sqrt(result_error);
}

void MeshSimplify::lock_positions() {
    // build position hasher
    auto position_getter = [&](uint32_t index) -> const Position & {
        return input_view.get_position_indexed(index);
    };
    PositionMultiMap<uint32_t, decltype(position_getter)> map(position_getter,
                                                              input_view.indices.size());

    for (auto i : input_view.indices) {
        map.emplace(i, i);
    }

    std::unordered_set<uint32_t> locked;

    for (auto i : locked_positions) {
        auto [begin, end] = map.equal_range(i);
        for (auto it = begin; it != end; ++it) {
            auto index = it->second;
            locked.emplace(index);
        }
    }

    locked_positions.resize(locked.size());
    std::copy(locked.begin(), locked.end(), locked_positions.begin());
}

void MeshSimplify::dump_obj(std::string_view filename) {
    std::ofstream file(filename.data());

    if (!file.is_open()) {
        std::cout << "failed to open file: " << filename << "\n";
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
