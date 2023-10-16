#include "builder.hpp"
#include "cluster.hpp"
#include "cluster_group.hpp"
#include "hasher.hpp"
#include "partitioner.hpp"
#include "simplify.hpp"
#include "utility.hpp"

#include <boost/graph/depth_first_search.hpp>

#include <boost/graph/detail/adjacency_list.hpp>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdint.h>

#include <typeindex>

template <typename PositionGetter>
void validate_cluster(const MeshDataView &                          view,
                      const EdgeMultiMap<uint32_t, PositionGetter> &edge_map, const AdjGraph &graph,
                      const MetisGraphData &metis_graph) {
    std::unordered_multimap<uint64_t, uint32_t> edge_map2(view.indices.size());

    for (uint32_t i = 0; i < view.indices.size(); i++) {
        uint64_t index1 = view.indices[i];
        uint64_t index2 = view.indices[triangle_cycle(i)];

        auto key = index1 << 32 | index2;

        edge_map2.emplace(key, i);
    }

    auto adj_edges2          = 0U;
    auto edge_map2_duplicate = 0U;
    for (uint32_t i = 0; i < view.indices.size(); i++) {
        uint64_t index1 = view.indices[i];
        uint64_t index2 = view.indices[triangle_cycle(i)];

        auto key          = index1 << 32 | index2;
        auto neighbor_key = index2 << 32 | index1;

        auto [begin, end] = edge_map2.equal_range(neighbor_key);
        auto count        = std::distance(begin, end);
        adj_edges2 += count;

        for (auto it = begin; it != end; ++it) {
            if (it->second == i) {
                edge_map2_duplicate++;
            }
        }
    }

    auto get_position = [&](auto edge_index) { return view.get_position_indexed(edge_index); };

    bool valid = true;

    uint32_t adj_edges1 = 0U;

    for (uint32_t i = 0; i < view.indices.size(); i++) {
        auto [begin, end] = edge_map.adjacent_range(i);

        if (begin == end) {
            continue;
        }

        adj_edges1 += std::distance(begin, end);

        auto pos1 = view.get_vertex(view.indices[i]);
        auto pos2 = view.get_vertex(view.indices[triangle_cycle(i)]);

        auto triangle_index = i / 3;
        for (auto it = begin; it != end; ++it) {
            auto adj_triangle_index = it->second / 3;
            if (adj_triangle_index == triangle_index) {
                continue;
            }

            auto [e, exist] = edge(triangle_index, adj_triangle_index, graph);

            if (!exist) {
                std::cout << "edge not exist" << std::endl;
                valid = false;
            }
        }
    }

    auto [vertex_begin, vertex_end] = boost::vertices(graph);
    for (auto it = vertex_begin; it != vertex_end; ++it) {
        auto vid   = *it;
        auto edges = boost::out_edges(vid, graph);
        auto count = std::distance(edges.first, edges.second);
        if (count != metis_graph.xadj[vid + 1] - metis_graph.xadj[vid]) {
            std::cout << "vertex " << vid << " edge count not match" << std::endl;
            valid = false;
        }
    }

    auto total_weight           = 0U;
    auto [edge_begin, edge_end] = boost::edges(graph);
    for (auto it = edge_begin; it != edge_end; ++it) {
        total_weight += get(boost::edge_weight, graph, *it);
    }

    auto metis_weight = std::accumulate(metis_graph.adjwgt.begin(), metis_graph.adjwgt.end(), 0U);

    auto degree = boost::num_edges(graph);

    // clang-format off
    std::cout << "[validate] num edges                              : " << view.indices.size() << std::endl;
    std::cout << "[validate] num adj edges in hash table2           : " << adj_edges2 << std::endl;
    std::cout << "[validate] num dup adj edges in hash table2       : " << edge_map2_duplicate << std::endl;
    std::cout << "[validate] num adj edges in hash table            : " << adj_edges1 << std::endl;
    std::cout << "[validate] num adj edges in graph                 : " << degree << std::endl;
    std::cout << "[validate] num adj edges in metis                 : " << metis_graph.xadj.back() << std::endl;
    std::cout << "[validate] total weight                           : " << total_weight << std::endl;
    std::cout << "[validate] metis weight                           : " << metis_weight << std::endl;
    // clang-format on

    if (!valid) {
        std::cout << "cluster is not valid" << std::endl;
        exit(1);
    }
}

void cluster_triangles(MeshDataView &view, std::vector<ClusterImpl> &clusters, const Bounds &bounds,
                       const SphereBounds &sphere_bounds) {
    // build edge hash map

    auto position_getter = [&](auto edge_index) { return view.get_position_indexed(edge_index); };

    EdgeMultiMap<uint32_t, decltype(position_getter)> edge_map(position_getter,
                                                               view.indices.size());

    {
        Timer timer("build edge map");
        for (auto i = 0U; i < view.indices.size(); i++) {
            edge_map.emplace(i, i);
        }
    }

    // build triangle adj data
    AdjGraph graph(view.indices.size() / 3);
    {
        Timer timer("build metis graph");
        build_triangle_adj_graph(view, edge_map, graph);
    }

    // convert boost graph to metis graph
    MetisGraphData graph_data;
    {
        Timer timer("convert boost graph to metis graph");
        graph_data = boost_graph_to_metis_graph(graph);
    }

#if _DEBUG
    {
        // validate cluster
        Timer timer("validate cluster");
        validate_cluster(view, edge_map, graph, graph_data);
    }
#endif

    // partition the graph
    const auto       num_triangles = view.indices.size() / 3;
    GraphPartitioner partitioner(num_triangles);
    {
        Timer timer("metis partitioning");
        // partitioner.directPartition(graph_data, Cluster::min_cluster_size,
        //                             Cluster::max_cluster_size);

        partitioner.bisectPartition(graph_data, ClusterImpl::min_cluster_size,
                                    ClusterImpl::max_cluster_size);
    }

    // build clusters
    clusters.reserve(partitioner.getNumClusters());
    {
        Timer timer("build clusters");
        for (auto i = 0U; i < partitioner.getNumClusters(); ++i) {
            auto partition = partitioner.ranges[i];

#if _DEBUG
            std::cout << "[DEBUG] cluster " << i << " range: [" << partition.first << ", "
                      << partition.second << "] count: " << partition.second - partition.first
                      << std::endl;
#endif
            auto &cluster = clusters.emplace_back();

            cluster.build_cluster(view, partitioner, partition.first, partition.second, edge_map);

            cluster.lodError = -1.0f; // cluster lod error is -1.0f for LOD0
        }
    }

    // ! if iterator debug level is 2 on windows, the destruction of boost graph will be super slow,
    // trust me, super slow, change it to 1 or debug using linux

    return;
}

void build(float *vertices, uint32_t num_vertices, std::vector<uint32_t> indices, bool has_color,
           bool has_tangent, uint32_t uv_count, BuilderSettings settings, DataStream &stream) {
    if (!settings.enable) {
        stream.buffer = {};
        return;
    }

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

    MeshDataView view(std::span(vertices, num_vertices * vertex_stride), num_vertices,
                      vertex_stride, std::span(indices.data(), indices.size()), attr_strides,
                      attr_weights);

    auto clusters       = std::vector<ClusterImpl>{};
    auto cluster_groups = std::vector<ClusterGroupImpl>{};

    auto mesh_bound  = Bounds{};
    auto mesh_sphere = SphereBounds{};

    cluster_triangles(view, clusters, mesh_bound, mesh_sphere);

    std::cout << "cluster building done" << std::endl;
    std::cout << "num clusters: " << clusters.size() << std::endl;

    build_cluster_dag(clusters, cluster_groups, 0, clusters.size());
    std::cout << "cluster dag building done" << std::endl;
    std::cout << "num cluster groups: " << cluster_groups.size() << std::endl;

    // Optional: hierarchy
    // after the clusters and cluster dag, we need to build acceleration structure to speed up
    // the culling process

    // save data to stream
    // serialization
    // TODO: important serialization

    return;
}