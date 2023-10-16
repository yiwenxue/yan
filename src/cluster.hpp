#pragma once

#include "hasher.hpp"
#include "partitioner.hpp"
#include "utility.hpp"

#include "builder.hpp"

#include <algorithm>
#include <bitset>
#include <boost/graph/detail/adjacency_list.hpp>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <ranges>
#include <span>
#include <stdint.h>
#include <unordered_map>
#include <utility>
#include <vector>

template <typename PositionGetter>
void build_triangle_adj_graph(const MeshDataView &                          view,
                              const EdgeMultiMap<uint32_t, PositionGetter> &edge_map,
                              AdjGraph &                                    graph);

struct ClusterImpl : public Cluster {
    static constexpr uint32_t min_cluster_size = 120;
    static constexpr uint32_t max_cluster_size = 128;

    MeshDataView view;

    std::vector<float>                vertex_array;
    std::vector<uint32_t>             index_array; // from pos0 to pos1, so it's also edge index
    std::bitset<max_cluster_size * 3> is_external; // only for real cluster data, for
                                                   // merged temp cluster, use external_edges
    std::vector<uint8_t> external_edges; // normal cluster is 0, only valid for temp clusters

    ClusterImpl() = default;

    // builder
    template <typename PositionGetter>
    void build_cluster(MeshDataView view, GraphPartitioner &partitioner, uint32_t tri_start,
                       uint32_t tri_end, EdgeMultiMap<uint32_t, PositionGetter> &edge_map) {
        auto num_triangles = tri_end - tri_start;

        vertex_array.reserve(num_triangles * view.vertex_stride);
        index_array.reserve(num_triangles * 3);
        external_edges.reserve(num_triangles * 3);

        std::unordered_map<uint32_t, uint32_t> oldToNewIndex; // new index for old ones will only be
                                                              // generated once

        for (auto i = tri_start; i < tri_end; i++) {
            auto triangle_index = partitioner.index[i];
            for (auto j = 0; j < 3; ++j) {
                auto old_index = view.indices[triangle_index * 3 + j];
                auto new_index = 0U;
                if (oldToNewIndex.find(old_index) != oldToNewIndex.end()) {
                    // if already converted, use the new index
                    new_index = oldToNewIndex[old_index];
                } else {
                    // if not converted, convert it and insert it into the map
                    new_index                = vertex_array.size() / view.vertex_stride;
                    oldToNewIndex[old_index] = new_index;
                    vertex_array.insert(vertex_array.end(), view.get_vertex(old_index),
                                        view.get_vertex(old_index) + view.vertex_stride);
                }

                index_array.emplace_back(new_index);

                auto edge_idx     = triangle_index * 3 + j;
                auto [begin, end] = edge_map.adjacent_range(edge_idx);

                auto new_edge_idx = index_array.size() - 1;

                // external deges:
                // 1, the edge is on the boundary of two adjacent clusters
                // ! the externals of lod0 is not seen as external, since it is doesn't belongs to
                // adjacent clusters

                // how many external edges does this edge have
                auto external_count = std::count_if(begin, end, [&](auto adj_edge_idx) {
                    auto triangle_id = partitioner.sortTo[adj_edge_idx.second / 3];
                    return triangle_id >= tri_end || triangle_id < tri_start;
                });
                external_edges.emplace_back(
                    external_count); // only for temp clusters use, but still need to record
                is_external[new_edge_idx] = external_count > 0;
            }
        }

        // construct MeshDataView
        this->view          = view;
        this->view.vertices = std::span(vertex_array);
        this->view.indices  = std::span(index_array);

        bound();
    }

    // merge constructor
    // ! TODO: input range error
    void merge_clusters(std::ranges::input_range auto &&clusters) {
        if (clusters.empty()) {
            return;
        }

        for (auto i = 0U; i < clusters.size(); i++) {
            assert(std::equal(clusters[i].view.attr_stride.begin(),
                              clusters[i].view.attr_stride.end(),
                              clusters[0].view.attr_stride.begin()));

            assert(std::equal(clusters[i].view.attr_weight.begin(),
                              clusters[i].view.attr_weight.end(),
                              clusters[0].view.attr_weight.begin()));

            assert(clusters[i].material_id == clusters[0].material_id);
        }

        const auto num_triangle_hint = std::accumulate(
            clusters.begin(), clusters.end(), 0ULL,
            [](auto sum, const auto &cluster) { return sum + cluster.view.indices.size() / 3; });

        vertex_array.reserve(num_triangle_hint * clusters[0].view.vertex_stride);
        index_array.reserve(num_triangle_hint * 3);
        // external, more than 128
        external_edges.reserve(num_triangle_hint * 3);

        lodLevel = (*std::max_element(clusters.begin(), clusters.end(),
                                      [](const auto &lhs, const auto &rhs) {
                                          return lhs.lodLevel < rhs.lodLevel;
                                      }))
                       .lodLevel;

        lodError = (*std::max_element(clusters.begin(), clusters.end(),
                                      [](const auto &lhs, const auto &rhs) {
                                          return lhs.lodError < rhs.lodError;
                                      }))
                       .lodError;

        std::unordered_multimap<uint32_t, uint32_t> positionMap;

        for (const auto &cluster : clusters) {
            for (auto i : cluster.index_array) {
                auto index = add_vertex(cluster.view.get_vertex(i), positionMap);
                index_array.emplace_back(index);
            }
            external_edges.insert(external_edges.end(), cluster.external_edges.begin(),
                                  cluster.external_edges.end());
        }

        // construct MeshDataView, attributes are the same as the first cluster, the vertex and
        // index are merged
        view          = clusters[0].view;
        view.vertices = std::span(vertex_array);
        view.indices  = std::span(index_array);

        // update the external edge count
        // external edges to skip the internal edges
        auto external_edges_index_view
            = std::views::iota(0U, external_edges.size())
              | std::views::filter([&](auto i) { return external_edges[i] > 0; });

        auto self_position_getter
            = [&](uint32_t index) -> const Position & { return view.get_position_indexed(index); };

        EdgeMultiMap<uint32_t, decltype(self_position_getter)> edge_map(self_position_getter,
                                                                        view.indices.size());

        // build adjacency map only for external edges, reasonable, since external edges are
        // detected by external count
        // ! if there exists an adj edge that is in another cluster, it must be external, and if it
        // is external edge, its adjacent edges (in other clusters) must be external too. this save
        // a lot of time
        for (auto i : external_edges_index_view) {
            edge_map.emplace(i, i);
        }

        uint32_t ChildIndex = 0;
        uint32_t MinIndex   = 0;
        uint32_t MaxIndex   = clusters[0].external_edges.size();

        for (auto i : external_edges_index_view) {
            if (i >= MaxIndex) {
                ChildIndex++;
                MinIndex = MaxIndex;
                MaxIndex += clusters[ChildIndex].external_edges.size();
            }

            auto [begin, end] = edge_map.adjacent_range(i);

            auto neighbot_count = std::count_if(begin, end, [&](auto it) {
                auto adj_edge_idx = it.second;
                // if th adjacency edge is found in other clusters in the merge list, the external
                // count should decrease by one
                return adj_edge_idx < MaxIndex || adj_edge_idx >= MinIndex;
            });

            // num of the adjacency neighbor edges must be smaller than the external count
            assert(external_edges[i] >= neighbot_count);
            external_edges[i] -= neighbot_count;
        }

        bound(); // calculate bounds
    }

    void simplify(uint32_t target_num_triangles, float target_error = 0.0f);

    template <typename PositionGetter>
    void split(GraphPartitioner &                            partitioner,
               const EdgeMultiMap<uint32_t, PositionGetter> &edge_map) {
        const auto num_triangles = view.indices.size() / 3;
        AdjGraph   graph(num_triangles);
        build_triangle_adj_graph(view, edge_map, graph);

        MetisGraphData graph_data = boost_graph_to_metis_graph(graph);

        partitioner.bisectPartition(graph_data, ClusterImpl::min_cluster_size,
                                    ClusterImpl::max_cluster_size);
        // finished partitioning, now we can split cluster according to the partition result
    }

    template <typename PositionGetter>
    EdgeMultiMap<uint32_t, PositionGetter> build_edge_map() {
        auto position_getter
            = [&](uint32_t index) -> const Position & { return view.get_position_indexed(index); };

        EdgeMultiMap<uint32_t, decltype(position_getter)> edge_map(position_getter,
                                                                   view.indices.size());
        for (auto i = 0U; i < view.indices.size(); i++) {
            edge_map.emplace(i, i);
        }

        return std::move(edge_map);
    }

    void clear();

private:
    uint32_t add_vertex(const float *vert, std::unordered_multimap<uint32_t, uint32_t> &table);

    void bound();
};

static inline std::iostream &operator<<(std::iostream &os, const Cluster &cluster);

void cluster_triangles(MeshDataView &view, std::vector<ClusterImpl> &clusters, const Bounds &bounds,
                       const SphereBounds &sphere_bounds);

template <typename PositionGetter>
void build_triangle_adj_graph(const MeshDataView &                          view,
                              const EdgeMultiMap<uint32_t, PositionGetter> &edge_map,
                              AdjGraph &                                    graph) {
    const auto num_triangles = view.indices.size() / 3;

    auto get_position = [&](auto edge_index) { return view.get_position_indexed(edge_index); };

    for (uint32_t i = 0; i < view.indices.size(); i++) {
        auto [begin, end] = edge_map.adjacent_range(i);

        if (begin == end) {
            continue;
        }

        auto triangle_index = i / 3;
        for (auto it = begin; it != end; ++it) {
            auto adj_triangle_index = it->second / 3;
            if (adj_triangle_index == triangle_index) {
                continue;
            }

            assert(triangle_index < num_triangles);
            assert(adj_triangle_index < num_triangles);

            auto [e, exist] = boost::edge(triangle_index, adj_triangle_index, graph);

            if (exist) {
                // increase weight
                put(boost::edge_weight, graph, e, get(boost::edge_weight, graph, e) + 1);
            } else {
                // init weight
                boost::add_edge(triangle_index, adj_triangle_index, 1U, graph);
            }
        }
    }

    return;
}
