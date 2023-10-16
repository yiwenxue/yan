#include "cluster_group.hpp"

static constexpr uint32_t merge_max_index_count = ClusterImpl::max_cluster_size * 6;

static void evaluate_cluster_group(std::vector<ClusterImpl> &     in_clusters,
                                   std::atomic<uint32_t> &        cluster_count,
                                   std::vector<ClusterGroupImpl> &groups, uint32_t group_id,
                                   std::span<uint32_t> children) {
    // the input range will construct a cluster group
    ClusterImpl merged = {};

    auto children_clusters
        = std::views::iota(0U, 100U)
          | std::views::transform([&](auto i) -> ClusterImpl & { return in_clusters[i]; });

    merged.merge_clusters(children_clusters);

    uint32_t parent_count
        = (merged.view.indices.size() + merge_max_index_count - 1) / merge_max_index_count;

    uint32_t parent_start = 0;
    uint32_t parent_end   = 0;

    for (uint32_t targetClusterSize = ClusterImpl::max_cluster_size;
         targetClusterSize >= ClusterImpl::min_cluster_size; targetClusterSize -= 2) {
        // one can not build the proper cluster directly, we need to tune the parameters until
        // we get the propr settings
        uint32_t targetTriangleCount = targetClusterSize * parent_count;

        merged.simplify(targetTriangleCount, 0.5);
        // the simplify error is stored in merged.lodError

        if (parent_count == 1) {
            // the last iteration, we need to save and then break
            parent_end   = (cluster_count += parent_count);
            parent_start = parent_end - parent_count;

            in_clusters[parent_start] = merged;
            break;
        } else {
            auto position_getter
                = [&](auto edge_index) { return merged.view.get_position_indexed(edge_index); };

            EdgeMultiMap<uint32_t, decltype(position_getter)> adjacency_map(
                position_getter, merged.view.indices.size());

            for (auto i : merged.view.indices) {
                adjacency_map.emplace(i, i);
            }

            GraphPartitioner partitioner(merged.view.indices.size() / 3);
            merged.split(partitioner, adjacency_map);

            if (partitioner.ranges.size() <= parent_count) {
                // we already get the proper one, we need to save and then break
                // generate new clusters

                parent_count = partitioner.getNumClusters();
                parent_end   = (cluster_count += parent_count);
                parent_start = parent_end - parent_count;

                for (auto i = 0U; i < parent_count; ++i) {
                    auto &range   = partitioner.ranges[i];
                    auto &cluster = in_clusters.at(i + parent_start);
                    cluster.build_cluster(merged.view, partitioner, range.first, range.second,
                                          adjacency_map);
                    cluster.lodError = merged.lodError; // inherit the error from parent
                    cluster.group_id = groups.size();
                }
                break;
            }
        }

        merged.merge_clusters(children_clusters);
    }

    // collect cluster group info
    auto &group = groups.at(group_id);

    float max_parent_error = 0.0f;
    float max_child_error  = 0.0f;
    float min_lod_error    = std::numeric_limits<float>::max();

    Bounds                    bounds_merged{};
    std::vector<SphereBounds> sphere_bounds(children.size());

    for (auto child : children) {
        auto &cluster = in_clusters[child];

        max_child_error = std::max(max_child_error, cluster.lodError);
        min_lod_error   = std::min(min_lod_error, cluster.lodError); // -1 for lod0
        group.children.emplace_back(child);
        cluster.group_id = group_id;

        bounds_merged = merge(bounds_merged, cluster.bounds);
        sphere_bounds.emplace_back(cluster.sphere_bounds);
    }

    //! min_child_error <= child_errors <= max_child_error <= parent_errors = max_parent_error

    // ensure that the parent cluster has error larger than the child cluster
    for (uint32_t i = parent_start; i < parent_end; ++i) {
        auto &cluster               = in_clusters[i];
        cluster.lodError            = max_child_error;
        max_parent_error            = std::max(max_parent_error, cluster.lodError);
        cluster.generating_group_id = group_id;
    }

    SphereBounds sphere_bounds_merged{};
    // sphere bounds from balls
    std::vector<SphereBounds> sphere_bounds_balls(children.size());
    for (auto child : children) {
        auto &cluster = in_clusters[child];
        sphere_bounds_balls.emplace_back(cluster.sphere_bounds);
    }
    sphere_bounds_merged = sphere_from_balls(sphere_bounds_balls);

    group.minLodError    = min_lod_error;
    group.maxParentError = max_parent_error;
    group.lodLevel       = merged.lodLevel + 1;
    group.bounds         = bounds_merged;
    group.sphere_bounds  = sphere_bounds_merged;
}

void build_cluster_dag(std::vector<ClusterImpl> &in_clusters, std::vector<ClusterGroupImpl> &groups,
                       uint32_t start, uint32_t count) {
    uint32_t levelOffset = start;

    bool isRootLevel = true;

    std::atomic<uint32_t> cluster_count{
        static_cast<uint32_t>(in_clusters.size())}; // for parallel sync

    while (true) {
        auto levelClusters
            = std::vector<uint32_t>(isRootLevel ? count : in_clusters.size() - levelOffset);

        isRootLevel = false;

        // fill the level clusters start from level offset
        std::iota(levelClusters.begin(), levelClusters.end(), levelOffset);

        if (levelClusters.size() < 2) {
            // it means that the clusters might be the root level, we need to exit the loop
            break;
        }

        if (levelClusters.size() < ClusterGroupImpl::max_group_size) {
            // the clusters are smaller enough, they can construct a cluster group directly
            uint32_t              parent_count = 0U;
            std::vector<uint32_t> children;
            for (auto child : levelClusters) {
                children.emplace_back(levelOffset++);
                parent_count += in_clusters[child].view.indices.size();
            }
            levelOffset = in_clusters.size();

            parent_count = (parent_count + merge_max_index_count - 1) / merge_max_index_count;
            in_clusters.resize(in_clusters.size() + parent_count);

            groups.emplace_back();
            evaluate_cluster_group(in_clusters, cluster_count, groups, groups.size() - 1,
                                   levelClusters);

            continue;
        }

        // for neither case, there are too many clusters, we need to partition them, merge, and
        // cluster ..
        // 1st, we need to build external edges map
        std::unordered_multimap<Edge, uint32_t, EdgeHasher> external_edge_map;
        for (auto cluster_id : levelClusters) {
            auto &cluster = in_clusters[cluster_id];
            for (auto i = 0U; i < cluster.external_edges.size(); ++i) {
                auto edge = cluster.external_edges[i];
                if (edge > 0) {
                    external_edge_map.emplace(
                        Edge{cluster.view.get_position_indexed(i),
                             cluster.view.get_position_indexed(triangle_cycle(i))},
                        cluster_id);
                }
            }
        }

        // 2nd, with external edge map, we can build the adjacency graph
        AdjGraph graph(levelClusters.size());
        for (auto [edge, cluster_id] : external_edge_map) {
            auto adj_edge     = Edge{edge.p1, edge.p0};
            auto [begin, end] = external_edge_map.equal_range(adj_edge);
            for (auto it = begin; it != end; ++it) {
                auto adj_cluster_id = it->second;
                if (adj_cluster_id == cluster_id) {
                    continue;
                }
                auto [e, exist] = boost::edge(cluster_id, adj_cluster_id, graph);
                if (exist) {
                    put(boost::edge_weight, graph, e, get(boost::edge_weight, graph, e) + 1);
                } else {
                    boost::add_edge(cluster_id, adj_cluster_id, 1U, graph);
                }
            }
        }

        // 3rd, convert boost graph to metis graph
        MetisGraphData graph_data = boost_graph_to_metis_graph(graph);

        // 4th, partition the graph
        GraphPartitioner partitioner(levelClusters.size());
        partitioner.bisectPartition(graph_data, ClusterImpl::min_cluster_size,
                                    ClusterImpl::max_cluster_size);

        uint32_t new_groups  = partitioner.ranges.size();
        uint32_t new_parents = 0u;
        for (auto &range : partitioner.ranges) {
            uint32_t indexCount = 0U;
            for (auto i = range.first; i < range.second; ++i) {
                partitioner.index[i] += levelOffset;
                indexCount += in_clusters[partitioner.index[i]].view.indices.size();
            }
            new_parents += ((indexCount + merge_max_index_count - 1) / merge_max_index_count);
        }

        levelOffset = in_clusters.size();

        // allocate new clusters and groups before dispatching the task, so that we can enable
        // parallel construction, suggest using unifex::static_task_pool or tbb task scheduler
        in_clusters.resize(in_clusters.size() + new_parents);
        groups.resize(groups.size() + new_groups);

        // this part can be parallelized, for each the parallel execute policy
        for (uint32_t i = 0u; i < partitioner.getNumClusters(); i++) {
            const auto &range = partitioner.ranges[i];

            std::vector<uint32_t> children(partitioner.index.begin() + range.first,
                                           partitioner.index.begin() + range.second);

            uint32_t group_index = groups.size() - partitioner.ranges.size() + i;
            evaluate_cluster_group(in_clusters, cluster_count, groups, group_index, children);
        }
    }

    auto &root = groups.emplace_back();
    // root group properties
    auto rootIndex                  = levelOffset;
    root.children                   = std::vector{rootIndex};
    root.bounds                     = in_clusters[rootIndex].bounds;
    root.sphere_bounds              = in_clusters[rootIndex].sphere_bounds;
    root.maxParentError             = std::numeric_limits<float>::max();
    root.minLodError                = -1.0f;
    root.lodLevel                   = in_clusters[rootIndex].lodLevel + 1;
    in_clusters[rootIndex].group_id = groups.size() - 1;

    return;
}
