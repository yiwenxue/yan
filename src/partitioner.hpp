#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include <metis.h>

#include <boost/graph/adjacency_list.hpp>

struct MetisGraphData {
    idx_t              nvtxs{0};
    std::vector<idx_t> xadj{};
    std::vector<idx_t> adjncy{};
    std::vector<idx_t> adjwgt{};
};

using AdjGraph
    = boost::adjacency_list<boost::listS,                                    // out edge list
                            boost::vecS,                                     // vertex list
                            boost::undirectedS,                              // undirected graph
                            boost::no_property,                              // vertex property
                            boost::property<boost::edge_weight_t, uint32_t>, // edge property
                            boost::disallow_parallel_edge_tag                // no parallel edge
                            >;

// convert boost graph to metis graph
template <typename Graph>
MetisGraphData boost_graph_to_metis_graph(const Graph &graph) {
    static_assert(std::is_integral_v<typename boost::graph_traits<Graph>::vertex_descriptor>,
                  "edge property type must be integral");

    MetisGraphData data;

    data.nvtxs        = boost::num_vertices(graph);
    const auto degree = boost::num_edges(graph) * 2;

    data.xadj.resize(data.nvtxs + 1);
    data.adjncy.resize(degree);
    data.adjwgt.resize(degree);

    for (uint32_t i = 0; i < data.nvtxs; ++i) {
        data.xadj[i]      = i == 0 ? 0 : data.xadj[i - 1]; // offset
        auto [begin, end] = boost::out_edges(i, graph);
        for (auto it = begin; it != end; ++it) {
            data.adjncy[data.xadj[i]] = boost::target(*it, graph);
            data.adjwgt[data.xadj[i]] = boost::get(boost::edge_weight, graph, *it);
            data.xadj[i]++;
        }
    }

    std::rotate(data.xadj.begin(), data.xadj.end() - 1, data.xadj.end());
    data.xadj[0] = 0;

    return std::move(data);
}

struct GraphPartitioner {
public:
    std::vector<uint32_t> sortTo;

    std::vector<uint32_t>                      index;  // index of the elements
    std::vector<std::pair<uint32_t, uint32_t>> ranges; // ranges of the clusters

    GraphPartitioner() = default;

    explicit GraphPartitioner(uint32_t num);

    // partition the graph directly into many parts
    void directPartition(MetisGraphData &graph, uint32_t min_cluster_size,
                         uint32_t max_cluster_size);

    // partition the graph through recursive bisection
    void bisectPartition(MetisGraphData &graph, uint32_t min_cluster_size,
                         uint32_t max_cluster_size);

    inline uint32_t getNumClusters() const {
        return ranges.size();
    }

    std::span<const uint32_t> getClusterContents(uint32_t i) const;

private:
    void recursive_bisection(MetisGraphData &graph, uint32_t start, uint32_t end);

    uint32_t bisection(MetisGraphData &graph, MetisGraphData &child1, MetisGraphData &child2,
                       uint32_t start, uint32_t end);

private:
    uint32_t num_elements;
    uint32_t max_cluster_size;
    uint32_t min_cluster_size;

    std::vector<idx_t> partition_id;

    std::vector<std::pair<uint32_t, uint32_t>> links; // not used for now
};
