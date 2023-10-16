#pragma once

#include "builder.hpp"
#include "hasher.hpp"
#include "partitioner.hpp"

#include <boost/graph/adjacency_list.hpp>

#include <chrono>
#include <iostream>
#include <stdint.h>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

class Timer {
public:
    Timer(std::string &&name) : name_(name), start_(std::chrono::steady_clock::now()) {
    }

    ~Timer() {
        const auto end = std::chrono::steady_clock::now();
        const auto duration
            = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        std::cout << name_ << " took " << duration << "ms" << std::endl;
    }

private:
    const std::string                           name_;
    const std::chrono::steady_clock::time_point start_;
};

// get the union of two bounds
static inline Bounds merge(const Bounds &a, const Bounds &b) {
    Bounds bounds;

    bounds.min[0] = std::min(a.min[0], b.min[0]);
    bounds.min[1] = std::min(a.min[1], b.min[1]);
    bounds.min[2] = std::min(a.min[2], b.min[2]);

    bounds.max[0] = std::max(a.max[0], b.max[0]);
    bounds.max[1] = std::max(a.max[1], b.max[1]);
    bounds.max[2] = std::max(a.max[2], b.max[2]);

    return bounds;
}

static inline Bounds extend(const Bounds &a, const float point[3]) {
    Bounds bounds;

    bounds.min[0] = std::fmin(a.min[0], point[0]);
    bounds.min[1] = std::fmin(a.min[1], point[1]);
    bounds.min[2] = std::fmin(a.min[2], point[2]);

    bounds.max[0] = std::fmax(a.max[0], point[0]);
    bounds.max[1] = std::fmax(a.max[1], point[1]);
    bounds.max[2] = std::fmax(a.max[2], point[2]);

    return bounds;
}

inline auto triangle_cycle(uint32_t i) -> uint32_t {
    return i % 3 == 0 ? i + 1 : (i % 3 == 1 ? i + 1 : i - 2);
}

struct Vec3 {
    union {
        struct {
            float x, y, z;
        };
        float data[3];
    };
};

inline Vec3 operator+(const Vec3 &lhs, const Vec3 &rhs) {
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

inline Vec3 operator-(const Vec3 &lhs, const Vec3 &rhs) {
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

inline Vec3 operator*(const Vec3 &lhs, float rhs) {
    return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
}

inline Vec3 operator*(float lhs, const Vec3 &rhs) {
    return rhs * lhs;
}

inline Vec3 operator*(const Vec3 &lhs, const Vec3 &rhs) {
    return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
}

inline Vec3 operator/(const Vec3 &lhs, float rhs) {
    return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
}

inline Vec3 operator/(float lhs, const Vec3 &rhs) {
    return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z};
}

inline Vec3 operator/(const Vec3 &lhs, const Vec3 &rhs) {
    return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z};
}

inline bool operator==(const Vec3 &lhs, const Vec3 &rhs) {
    return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
}

inline bool operator!=(const Vec3 &lhs, const Vec3 &rhs) {
    return !(lhs == rhs);
}

inline float dot(const Vec3 &lhs, const Vec3 &rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

inline float length(const Vec3 &v) {
    return std::sqrt(dot(v, v));
}

inline Vec3 normalize(const Vec3 &v) {
    auto l = length(v);
    if (l == 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    } else {
        return v / l;
    }
}

inline Vec3 cross(const Vec3 &lhs, const Vec3 &rhs) {
    return {lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z,
            lhs.x * rhs.y - lhs.y * rhs.x};
}

inline Vec3 lerp(const Vec3 &lhs, const Vec3 &rhs, float t) {
    return {
        std::lerp(lhs.x, rhs.x, t),
        std::lerp(lhs.y, rhs.y, t),
        std::lerp(lhs.z, rhs.z, t),
    };
}

using Position = Vec3;

struct Edge {
    Position p0, p1;
};

inline bool operator==(const Edge &lhs, const Edge &rhs) {
    return (lhs.p0 == rhs.p0) && (lhs.p1 == rhs.p1);
}

struct PositionHasher {
    size_t operator()(const Position &p) const {
        union {
            float    f;
            uint32_t i;
        } x, y, z;
        x.f = p.x;
        y.f = p.y;
        z.f = p.z;

        return murmur32({
            p.x == 0.0f ? 0u : x.i,
            p.y == 0.0f ? 0u : y.i,
            p.z == 0.0f ? 0u : z.i,
        });
    }
};

struct EdgeHasher {
    size_t operator()(const Edge &e) const {
        uint32_t hash1 = PositionHasher{}(e.p0);
        uint32_t hash2 = PositionHasher{}(e.p1);
        return murmur32({hash1, hash2});
    }
};

// Find edge with opposite direction that shares these 2 verts.
/*
      /\
     /  \
    o-<<-o
    o->>-o
     \  /
      \/
*/
struct EdgeHashTable {
    EdgeHashTable(uint32_t size) : table(size) {
    }

    using HashTable = std::unordered_multimap<Edge, uint32_t, EdgeHasher>;

    HashTable table;

    template <typename PositionGetter>
    requires std::is_invocable_r<const Position &, PositionGetter, uint32_t>::value void
        add_edge(uint32_t edgeIndex, PositionGetter &&positionGetter) {
        const auto &p1 = positionGetter(edgeIndex);
        const auto &p2 = positionGetter(triangle_cycle(edgeIndex));

        Edge edge{p1, p2};

        table.emplace(edge, edgeIndex);
    }

    template <typename PositionGetter>
    requires std::is_invocable_r<const Position &, PositionGetter, uint32_t>::value
        std::pair<HashTable::iterator, HashTable::iterator>
        get_adj(uint32_t edgeIndex, PositionGetter &&positionGetter) {
        const auto &p1 = positionGetter(edgeIndex);
        const auto &p2 = positionGetter(triangle_cycle(edgeIndex));

        Edge edge{p2, p1};

        return table.equal_range(edge);
    }

    template <typename PositionGetter>
    requires std::is_invocable_r<const Position &, PositionGetter, uint32_t>::value
        std::pair<HashTable::const_iterator, HashTable::const_iterator>
        get_adj(uint32_t edgeIndex, PositionGetter &&positionGetter)
    const {
        const auto &p1 = positionGetter(edgeIndex);
        const auto &p2 = positionGetter(triangle_cycle(edgeIndex));

        Edge edge{p2, p1};

        return table.equal_range(edge);
    }
};

template <typename T, typename PositionGetter, bool isMulti>
struct PositionHashMap {
    PositionHashMap(PositionGetter getter, uint32_t size = 0) :
        map(size, HashAdapter(getter), EqualAdapter(getter)) {
    }

    struct HashAdapter {
        PositionGetter getter;

        HashAdapter(PositionGetter getter) : getter(getter){};
        HashAdapter(const HashAdapter &) = default;
        HashAdapter(HashAdapter &&)      = default;
        HashAdapter &operator=(const HashAdapter &) = default;
        HashAdapter &operator=(HashAdapter &&) = default;
        ~HashAdapter()                         = default;

        size_t operator()(uint32_t i) const {
            return PositionHasher()(getter(i));
        }
    };

    struct EqualAdapter {
        PositionGetter getter;

        EqualAdapter(PositionGetter getter) : getter(getter){};
        EqualAdapter(const EqualAdapter &) = default;
        EqualAdapter(EqualAdapter &&)      = default;
        EqualAdapter &operator=(const EqualAdapter &) = default;
        EqualAdapter &operator=(EqualAdapter &&) = default;
        ~EqualAdapter()                          = default;

        bool operator()(const uint32_t l, const uint32_t r) const {
            return getter(l) == getter(r);
        }
    };

    // emplace
    template <typename... Args>
    auto emplace(uint32_t i, Args &&...args) {
        return map.emplace(std::piecewise_construct, std::make_tuple(i),
                           std::forward_as_tuple(args...));
    }

    // find
    auto find(uint32_t i) {
        return map.find(i);
    }

    const auto find(uint32_t i) const {
        return map.find(i);
    }

    // equal range
    auto equal_range(uint32_t i) {
        return map.equal_range(i);
    }

    const auto equal_range(uint32_t i) const {
        return map.equal_range(i);
    }

    auto at(uint32_t i) {
        return map.at(i);
    }

    const auto at(uint32_t i) const {
        return map.at(i);
    }

    auto count(uint32_t i) const {
        return map.count(i);
    }

    using HashMap
        = std::conditional_t<isMulti,
                             std::unordered_multimap<uint32_t, T, HashAdapter, EqualAdapter>,
                             std::unordered_map<uint32_t, T, HashAdapter, EqualAdapter>>;
    HashMap map;
};

template <typename T, typename PositionGetter = std::function<const Position &(uint32_t)>>
using PositionMap = PositionHashMap<T, PositionGetter, false>;

template <typename T, typename PositionGetter = std::function<const Position &(uint32_t)>>
using PositionMultiMap = PositionHashMap<T, PositionGetter, true>;

template <typename T, typename PositionGetter, bool isMulti>
struct EdgeHashMap {
    struct EdgeIndex {
        union {
            struct {
                uint32_t i0, i1;
            };
            uint64_t data;
        };
    };

    struct HashAdapter {
        PositionGetter getter;

        HashAdapter(PositionGetter getter) : getter(getter){};
        HashAdapter(const HashAdapter &) = default;
        HashAdapter(HashAdapter &&)      = default;
        HashAdapter &operator=(const HashAdapter &) = default;
        HashAdapter &operator=(HashAdapter &&) = default;
        ~HashAdapter()                         = default;

        size_t operator()(EdgeIndex i) const {
            return EdgeHasher()({
                getter(i.i0),
                getter(i.i1),
            });
        }
    };

    struct EqualAdapter {
        PositionGetter getter;

        EqualAdapter(PositionGetter getter) : getter(getter){};
        EqualAdapter(const EqualAdapter &) = default;
        EqualAdapter(EqualAdapter &&)      = default;
        EqualAdapter &operator=(const EqualAdapter &) = default;
        EqualAdapter &operator=(EqualAdapter &&) = default;
        ~EqualAdapter()                          = default;

        bool operator()(const EdgeIndex l, const EdgeIndex r) const {
            return getter(l.i0) == getter(r.i0) && getter(l.i1) == getter(r.i1);
        }
    };

    using HashMap
        = std::conditional_t<isMulti,
                             std::unordered_multimap<EdgeIndex, T, HashAdapter, EqualAdapter>,
                             std::unordered_map<EdgeIndex, T, HashAdapter, EqualAdapter>>;

    HashMap map;

    EdgeHashMap(PositionGetter getter, uint32_t size = 0) :
        map(size, HashAdapter(getter), EqualAdapter(getter)) {
    }

    // emplace
    template <typename... Args>
    auto emplace(uint32_t i, Args &&...args) {
        return map.emplace(std::piecewise_construct,
                           std::forward_as_tuple(EdgeIndex{i, triangle_cycle(i)}),
                           std::forward_as_tuple(args...));
    }

    // find
    auto find(uint32_t i) {
        return map.find({i, triangle_cycle(i)});
    }

    const auto find(uint32_t i) const {
        return map.find({i, triangle_cycle(i)});
    }

    // equal range
    auto equal_range(uint32_t i) {
        return map.equal_range({i, triangle_cycle(i)});
    }

    const auto equal_range(uint32_t i) const {
        return map.equal_range({i, triangle_cycle(i)});
    }

    // adjacent range
    auto adjacent_range(uint32_t i) {
        return map.equal_range({triangle_cycle(i), i});
    }

    const auto adjacent_range(uint32_t i) const {
        return map.equal_range({triangle_cycle(i), i});
    }
};

template <typename T, typename PositionGetter = std::function<const Position &(uint32_t)>>
using EdgeMap = EdgeHashMap<T, PositionGetter, false>;

template <typename T, typename PositionGetter = std::function<const Position &(uint32_t)>>
using EdgeMultiMap = EdgeHashMap<T, PositionGetter, true>;

/**
 * @brief simple MeshData view
 * 1. vertices are stored in a single array
 * 2. each vertex has a fixed size
 * 3. each vertex has position of 3 floats at the beginning
 * 4. attributes are stored after position
 */
struct MeshDataView {
    MeshDataView() = default;

    MeshDataView(std::span<float> vertices, uint32_t vertex_count,
                 uint32_t            vertex_stride,                            // unm of floats
                 std::span<uint32_t> indices, std::span<uint32_t> attr_stride, // num of floats
                 std::span<float> attr_weight) :
        vertices(vertices),
        vertex_count(vertex_count),
        vertex_stride(vertex_stride),
        indices(indices),
        attr_stride(attr_stride),
        attr_weight(attr_weight) {
        assert(std::accumulate(attr_stride.begin(), attr_stride.end(), 3u) == vertex_stride);
        attr_offset.resize(attr_stride.size());
        attr_offset[0] = 3u; // position
        for (int i = 1; i < attr_stride.size(); ++i) {
            attr_offset[i] = attr_offset[i - 1] + attr_stride[i - 1];
        }
    }

    MeshDataView(const MeshDataView &) = default;
    MeshDataView(MeshDataView &&)      = default;
    MeshDataView &operator=(const MeshDataView &) = default;
    MeshDataView &operator=(MeshDataView &&) = default;
    ~MeshDataView()                          = default;

    [[nodiscard]] const Position &get_position(uint32_t index) const {
        return *reinterpret_cast<const Position *>(vertices.data() + index * vertex_stride);
    }

    [[nodiscard]] Position &get_position(uint32_t index) {
        return *reinterpret_cast<Position *>(vertices.data() + index * vertex_stride);
    }

    [[nodiscard]] const Position &get_position_indexed(uint32_t index) const {
        return *reinterpret_cast<const Position *>(vertices.data()
                                                   + indices[index] * vertex_stride);
    }

    [[nodiscard]] Position &get_position_indexed(uint32_t index) {
        return *reinterpret_cast<Position *>(vertices.data() + indices[index] * vertex_stride);
    }

    [[nodiscard]] const float *get_vertex(uint32_t index) const {
        return vertices.data() + index * vertex_stride;
    }

    [[nodiscard]] float *get_vertex(uint32_t index) {
        return vertices.data() + index * vertex_stride;
    }

    [[nodiscard]] const float *get_vertex_indexed(uint32_t index) const {
        return vertices.data() + indices[index] * vertex_stride;
    }

    [[nodiscard]] float *get_vertex_indexed(uint32_t index) {
        return vertices.data() + indices[index] * vertex_stride;
    }

    [[nodiscard]] const float *get_attr(uint32_t index) const {
        return vertices.data() + index * vertex_stride + attr_offset[0];
    }

    [[nodiscard]] float *get_attr(uint32_t index) {
        return vertices.data() + index * vertex_stride + attr_offset[0];
    }

    [[nodiscard]] const float *get_attr_indexed(uint32_t index) const {
        return vertices.data() + indices[index] * vertex_stride + attr_offset[0];
    }

    [[nodiscard]] float *get_attr_indexed(uint32_t index) {
        return vertices.data() + indices[index] * vertex_stride + attr_offset[0];
    }

    void set_attribute(std::span<uint32_t> stride, std::span<float> weight) {
        attr_stride = stride;
        attr_weight = weight;
        for (int i = 1; i < attr_stride.size(); ++i) {
            attr_offset[i] = attr_offset[i - 1] + attr_stride[i - 1];
        }
    }

    std::span<float>    vertices;
    uint32_t            vertex_count;
    uint32_t            vertex_stride; // num of floats
    std::span<uint32_t> indices;

    std::span<uint32_t>   attr_stride; // num of floats
    std::span<float>      attr_weight;
    std::vector<uint32_t> attr_offset; // num of floats
};

template <typename Graph>
Graph metis_graph_to_boost(const MetisGraphData &data) {
    Graph graph;

    // add vertices
    for (uint32_t i = 0; i < data.nvtxs; ++i) {
        boost::add_vertex(graph);
    }

    // add edges
    for (uint32_t i = 0; i < data.nvtxs; ++i) {
        for (idx_t j = data.xadj[i]; j < data.xadj[i + 1]; ++j) {
            auto [edge, res] = boost::add_edge(i, data.adjncy[j], graph);
            boost::put(boost::edge_weight, graph, edge, data.adjwgt[j]);
        }
    }

    return std::move(graph);
}

Bounds get_bound(const MeshDataView &view);

SphereBounds sphere_from_balls(std::span<const SphereBounds> balls);

SphereBounds get_sphere_bound(const MeshDataView &view);