#pragma once

#include "builder.hpp"

#include "cluster.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

struct ClusterGroupImpl : public ClusterGroup {
    static constexpr uint32_t max_group_size = 32;
};

static inline std::iostream &operator<<(std::iostream &os, const ClusterGroup &group);

void build_cluster_dag(std::vector<ClusterImpl> &in_clusters, std::vector<ClusterGroupImpl> &groups,
                       uint32_t start, uint32_t count);