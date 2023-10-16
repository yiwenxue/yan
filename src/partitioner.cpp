#include "partitioner.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <metis.h>
#include <numeric>
#include <stdexcept>
#include <stdio.h>

GraphPartitioner::GraphPartitioner(uint32_t num) :
    num_elements{num}, max_cluster_size{num / 2}, min_cluster_size{num / 2} {
}

std::span<const uint32_t> GraphPartitioner::getClusterContents(uint32_t i) const {
    auto &range = ranges[i];
    return std::span<const uint32_t>(index).subspan(range.first, range.second - range.first);
}

void GraphPartitioner::directPartition(MetisGraphData &graph, uint32_t min_cluster_size,
                                       uint32_t max_cluster_size) {
    this->min_cluster_size = min_cluster_size;
    this->max_cluster_size = max_cluster_size;
    this->num_elements     = graph.nvtxs;

    const uint32_t target_partition_size = (min_cluster_size + max_cluster_size) / 2;
    const uint32_t target_num_partitions
        = (graph.nvtxs + target_partition_size - 1) / target_partition_size;

    index.resize(num_elements);
    partition_id.resize(num_elements);
    sortTo.resize(num_elements);
    std::iota(index.begin(), index.end(), 0U);

    if (target_num_partitions > 1) {
        links.clear();

        idx_t num_constraints = 1;
        idx_t num_parts       = target_num_partitions;
        idx_t edge_cuts       = 0;

        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_SEED]   = 31; //! fix seed for reproducibility
        options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_RANDOM;

        int res = METIS_PartGraphKway(&graph.nvtxs, &num_constraints, graph.xadj.data(),
                                      graph.adjncy.data(), nullptr, nullptr, graph.adjwgt.data(),
                                      &num_parts, nullptr, nullptr, options, &edge_cuts,
                                      partition_id.data());

        if (res != METIS_OK) {
            throw std::runtime_error("METIS partitioning failed");
        }

        auto partition_count = std::vector<uint32_t>(target_num_partitions, 0);
        for (auto i : partition_id) {
            partition_count[i]++;
        }

        // ranges
        ranges.resize(target_num_partitions); // resize for index convenience
        auto begin = 0U;
        for (auto i = 0U; i < target_num_partitions; i++) {
            ranges[i] = std::make_pair(begin, begin + partition_count[i]);
            begin += partition_count[i];
        }
        std::fill(partition_count.begin(), partition_count.end(), 0);

        // index
        for (auto i = 0U; i < num_elements; i++) {
            const auto partition_idx = partition_id[i];
            const auto offset        = ranges[partition_idx].first;
            const auto idx           = partition_count[partition_idx]++;

            index[offset + idx] = i;
        }
        partition_id.clear();
    } else {
        ranges.emplace_back(0, num_elements);
    }

    // sort to
    for (uint32_t i = 0; i < num_elements; i++) {
        sortTo[index[i]] = i;
    }
}

void GraphPartitioner::bisectPartition(MetisGraphData &graph, uint32_t min_cluster_size,
                                       uint32_t max_cluster_size) {
    this->min_cluster_size = min_cluster_size;
    this->max_cluster_size = max_cluster_size;
    this->num_elements     = graph.nvtxs;

    const uint32_t target_partition_size = (min_cluster_size + max_cluster_size) / 2;
    const uint32_t target_num_partitions
        = (graph.nvtxs + target_partition_size - 1) / target_partition_size;

    index.resize(num_elements);
    sortTo.resize(num_elements);
    partition_id.resize(num_elements);
    ranges.reserve(target_num_partitions); // for emplace_back convenience

    std::iota(index.begin(), index.end(), 0U);

    if (target_num_partitions > 1) {
        links.clear(); // TODO links
        recursive_bisection(graph, 0, graph.nvtxs);
    } else {
        ranges.emplace_back(0, num_elements);
    }

    // sort to
    for (uint32_t i = 0; i < num_elements; i++) {
        sortTo[index[i]] = i;
    }
}

void GraphPartitioner::recursive_bisection(MetisGraphData &graph, uint32_t start, uint32_t end) {
    MetisGraphData children[2];

    auto split = bisection(graph, children[0], children[1], start, end);

    if (children[0].nvtxs > 0 && children[1].nvtxs > 0) {
        recursive_bisection(children[0], start, split);
        recursive_bisection(children[1], split, end);
    }
}

uint32_t GraphPartitioner::bisection(MetisGraphData &graph, MetisGraphData &child1,
                                     MetisGraphData &child2, uint32_t start, uint32_t end) {
    assert(start < end);
    assert(end - start == graph.nvtxs);

    if (graph.nvtxs <= max_cluster_size) {
        ranges.emplace_back(start, end);
        return end;
    }

    const auto target_partition_size = (min_cluster_size + max_cluster_size) / 2;
    const auto target_num_partitions
        = std::max(std::lround(static_cast<float>(graph.nvtxs) / target_partition_size + 0.5f), 2L);

    auto first_partition_weight
        = static_cast<float>(target_num_partitions / 2) / target_num_partitions;

    float partition_weight[2] = {
        first_partition_weight,
        1.0f - first_partition_weight,
    };

    bool loose     = target_partition_size >= 128 || max_cluster_size / min_cluster_size > 1;
    bool fine      = graph.nvtxs <= 4096;
    bool superFine = graph.nvtxs <= 1024;

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_SEED]    = 31; //! fix seed for reproducibility
    options[METIS_OPTION_IPTYPE]  = METIS_IPTYPE_RANDOM;
    options[METIS_OPTION_NCUTS]   = superFine ? 8 : fine ? 4 : 1;
    options[METIS_OPTION_NITER]   = superFine ? 40 : fine ? 20 : 10;
    options[METIS_OPTION_UFACTOR] = loose ? 200 : 1;
    options[METIS_OPTION_MINCONN] = 1;

    std::span index_l        = std::span(index).subspan(start, end - start);
    std::span sortTo_l       = std::span(sortTo).subspan(start, end - start);
    std::span partition_id_l = std::span(partition_id).subspan(start, end - start);

    idx_t num_constraints = 1;
    idx_t edge_cuts       = 0;
    idx_t num_parts       = 2;

    int res = METIS_PartGraphRecursive(&graph.nvtxs, &num_constraints, graph.xadj.data(),
                                       graph.adjncy.data(), nullptr, nullptr, graph.adjwgt.data(),
                                       &num_parts, partition_weight, nullptr, options, &edge_cuts,
                                       partition_id_l.data());

    if (res != METIS_OK) {
        throw std::runtime_error("METIS partitioning failed");
    }

    std::iota(sortTo_l.begin(), sortTo_l.end(), 0);
    std::stable_sort(sortTo_l.begin(), sortTo_l.end(),
                     [&](uint32_t a, uint32_t b) { return partition_id_l[a] < partition_id_l[b]; });
    auto split     = std::stable_partition(sortTo_l.begin(), sortTo_l.end(),
                                       [&](uint32_t a) { return partition_id_l[a] == 0; });
    auto split_idx = std::distance(sortTo_l.begin(), split);

    uint32_t children_size[2] = {
        static_cast<uint32_t>(split_idx),
        static_cast<uint32_t>(graph.nvtxs - split_idx),
    };

    assert(children_size[0] > 1);
    assert(children_size[1] > 1);

    // partition id is now the origin index,
    std::copy(index_l.begin(), index_l.end(), partition_id_l.begin());

    // index is now the new index
    std::transform(sortTo_l.begin(), sortTo_l.end(), index_l.begin(),
                   [&](uint32_t a) { return partition_id_l[a]; });

    std::copy(sortTo_l.begin(), sortTo_l.end(), partition_id_l.begin());

    for (uint32_t i = 0; i < sortTo_l.size(); i++) {
        const auto &id = partition_id_l[i];
        sortTo_l[id]   = i;
    }

    if (children_size[0] <= max_cluster_size && children_size[1] <= max_cluster_size) {
        ranges.emplace_back(start, static_cast<uint32_t>(start + split_idx));
        ranges.emplace_back(static_cast<uint32_t>(start + split_idx), end);
    } else {
        // generate 2 children
        auto degree_est = graph.adjncy.size() >> 1U;

        child1.nvtxs = children_size[0];
        child1.xadj.reserve(child1.nvtxs + 1);
        child1.adjncy.reserve(degree_est);
        child1.adjwgt.reserve(degree_est);

        child2.nvtxs = children_size[1];
        child2.xadj.reserve(child2.nvtxs + 1);
        child2.adjncy.reserve(degree_est);
        child2.adjwgt.reserve(degree_est);

        for (auto i = 0U; i < graph.nvtxs; i++) {
            auto triangle = sortTo_l[i];
            if (triangle < split_idx) { // if i < split_index, the triangle is in the first half
                child1.xadj.push_back(child1.adjncy.size());
                // iterate over all neighbors
                for (auto j = graph.xadj[i]; j < graph.xadj[i + 1]; j++) {
                    // the neighbor triangle index in the old graph
                    auto old_neighbor = graph.adjncy[j];
                    // the neighbor triangle index in the new graph
                    auto neighbor = sortTo_l[old_neighbor];
                    if (neighbor < split_idx) { // if the neighbor is in the first half
                        // add new neighbor (neighbor index in the new graph)
                        child1.adjncy.push_back(neighbor);
                        // add new edge weight (weight index unchanged)
                        child1.adjwgt.push_back(graph.adjwgt[j]);
                    }
                }
            } else { // if i >= split_index, the triangle is in the second half
                child2.xadj.push_back(child2.adjncy.size());
                // iterate over all neighbors
                for (auto j = graph.xadj[i]; j < graph.xadj[i + 1]; j++) {
                    auto old_neighbor = graph.adjncy[j];
                    auto neighbor     = sortTo_l[old_neighbor];
                    if (neighbor >= split_idx) { // if the neighbor is in the second half
                        child2.adjncy.push_back(neighbor - split_idx);
                        child2.adjwgt.push_back(graph.adjwgt[j]);
                    }
                }
            }
        }
        child1.xadj.push_back(child1.adjncy.size());
        child2.xadj.push_back(child2.adjncy.size());

        assert(child1.xadj.size() == child1.nvtxs + 1);
        assert(child2.xadj.size() == child2.nvtxs + 1);
    }

    graph.xadj.clear();
    graph.adjncy.clear();
    graph.adjwgt.clear();

    assert(split_idx + start < end);

    return split_idx + start;
}
