#include "cluster.hpp"
#include "partitioner.hpp"
#include "simplify.hpp"
#include "utility.hpp"
#include <algorithm>

#include "simplify.hpp"

void ClusterImpl::bound() {
    // calculate bounds
    bounds        = get_bound(view);
    sphere_bounds = get_sphere_bound(view);
}

uint32_t ClusterImpl::add_vertex(const float *                                vert,
                                 std::unordered_multimap<uint32_t, uint32_t> &table) {
    const auto &position = *reinterpret_cast<const Position *>(vert);
    uint32_t    hash     = PositionHasher()(position);
    auto [begin, end]    = table.equal_range(hash);
    auto index           = 0xFFFFFFFF;
    for (auto it = begin; it != end; ++it) {
        if (std::equal(vert, vert + view.vertex_stride,
                       vertex_array.data() + it->second * view.vertex_stride)) {
            index = it->second;
            break;
        }
    }
    if (table.find(index) == table.end()) {
        index = vertex_array.size() / view.vertex_stride;
        vertex_array.insert(vertex_array.end(), vert, vert + view.vertex_stride);
        table.emplace(hash, index);
    }

    return index;
}

void ClusterImpl::simplify(uint32_t target_num_triangles, float target_error) {
    auto simplifier = MeshSimplify(view, true); // inplace simplify

    auto external_edges_index_view
        = std::views::iota(0U, external_edges.size())
          | std::views::filter([&](auto i) { return external_edges[i] > 0; });

    // lock all external edges
    for (auto i : external_edges_index_view) {
        simplifier.lock_position(i);
    }

    lodError = simplifier.simplify(target_num_triangles, target_error);

    bound(); // calculate bounds
}

void ClusterImpl::clear() {
    vertex_array.clear();
    index_array.clear();
    external_edges.clear();
    view = MeshDataView();
}