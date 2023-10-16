#include "simplifier.hpp"
#include "utility.hpp"
#include <algorithm>
#include <boost/graph/detail/adjacency_list.hpp>
#include <cmath>
#include <fstream>
#include <numeric>
#include <ranges>
#include <stdint.h>
#include <type_traits>
#include <unordered_map>

#include <cstdio>

// clang-format off
const unsigned char MeshSimplifier::kCanCollapse[MeshSimplifier::VertexKindCount]
                                                [MeshSimplifier::VertexKindCount] =
{
    {1, 1, 1, 1, 1},
    {0, 1, 0, 0, 0},
    {0, 0, 1, 0, 0},
    {0, 0, 0, 1, 1},
    {0, 0, 0, 0, 0},
};

const unsigned char MeshSimplifier::kHasOpposite[MeshSimplifier::VertexKindCount]
                                                [MeshSimplifier::VertexKindCount] =
{
    {1, 1, 1, 0, 1},
    {1, 0, 1, 0, 0},
    {1, 1, 1, 0, 1},
    {0, 0, 0, 0, 0},
    {1, 0, 1, 0, 0},
};
// clang-format on

float MeshSimplifier::Quadric::operator()(const Position &p) const {
    float rx = b0;
    float ry = b1;
    float rz = b2;

    rx += a10 * p.y;
    ry += a21 * p.z;
    rz += a20 * p.x;

    rx *= 2;
    ry *= 2;
    rz *= 2;

    rx += a00 * p.x;
    ry += a11 * p.y;
    rz += a22 * p.z;

    float r = c;
    r += rx * p.x;
    r += ry * p.y;
    r += rz * p.z;

    float s = w == 0.f ? 0.f : 1.f / w;

    return fabsf(r) * s;
}

MeshSimplifier::Quadric
    MeshSimplifier::Quadric::operator+(const MeshSimplifier::Quadric &rhs) const {
    return {
        a00 + rhs.a00, a11 + rhs.a11, a22 + rhs.a22, a10 + rhs.a10, a20 + rhs.a20, a21 + rhs.a21,
        b0 + rhs.b0,   b1 + rhs.b1,   b2 + rhs.b2,   c + rhs.c,     w + rhs.w,
    };
}

MeshSimplifier::Quadric &MeshSimplifier::Quadric::operator+=(const MeshSimplifier::Quadric &rhs) {
    a00 += rhs.a00;
    a11 += rhs.a11;
    a22 += rhs.a22;
    a10 += rhs.a10;
    a20 += rhs.a20;
    a21 += rhs.a21;
    b0 += rhs.b0;
    b1 += rhs.b1;
    b2 += rhs.b2;
    c += rhs.c;
    w += rhs.w;

    return *this;
}

void MeshSimplifier::Quadric::fromPoint(const Position &p, float weight) {
    w   = weight;
    a00 = w;
    a11 = w;
    a22 = w;
    a10 = 0.f;
    a20 = 0.f;
    a21 = 0.f;
    b0  = -2.f * p.x * w;
    b1  = -2.f * p.y * w;
    b2  = -2.f * p.z * w;
    c   = (p.x * p.x + p.y * p.y + p.z * p.z) * w;
}

void MeshSimplifier::Quadric::fromPlane(const Position &n, float d, float weight) {
    w = weight;

    float aw = n.x * w;
    float bw = n.y * w;
    float cw = n.z * w;
    float dw = d * w;

    a00 = n.x * aw;
    a11 = n.y * bw;
    a22 = n.z * cw;
    a10 = n.x * bw;
    a20 = n.x * cw;
    a21 = n.y * cw;
    b0  = n.x * dw;
    b1  = n.y * dw;
    b2  = n.z * dw;
    c   = d * dw;
}

void MeshSimplifier::Quadric::fromTriangle(const Position &a, const Position &b, const Position &c,
                                           float weight) {
    auto  v0   = b - a;
    auto  v1   = c - a;
    auto  n    = cross(v0, v1);
    float area = length(n);
    n          = normalize(n);
    auto d     = -dot(n, a);

    fromPlane(n, d, std::sqrt(weight) * area);
}

void MeshSimplifier::Quadric::fromTriangleEdge(const Position &a, const Position &b,
                                               const Position &c, float weight) {
    auto v0 = b - a;
    auto l  = length(v0);
    v0      = normalize(v0);

    auto v1 = c - a;
    auto h  = dot(v0, v1);

    // implement is important
}

void MeshSimplifier::Quadric::fromAttributes(
    std::span<MeshSimplifier::QuadricGrad> grads, // one per float
    const Position &p0, const Position &p1, const Position &p2, const float *attr0,
    const float *attr1, const float *attr2) {
    auto v0 = p1 - p0;
    auto v1 = p2 - p0;

    auto n    = cross(v0, v1);
    auto area = length(n);
    auto w    = std::sqrt(area);

    float d00    = v0.x * v0.x + v0.y * v0.y + v0.z * v0.z;
    float d01    = v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
    float d11    = v1.x * v1.x + v1.y * v1.y + v1.z * v1.z;
    float denom  = d00 * d11 - d01 * d01;
    float denomr = denom == 0 ? 0.f : 1.f / denom;

    float gx1 = (d11 * v0.x - d01 * v1.x) * denomr;
    float gx2 = (d00 * v1.x - d01 * v0.x) * denomr;
    float gy1 = (d11 * v0.y - d01 * v1.y) * denomr;
    float gy2 = (d00 * v1.y - d01 * v0.y) * denomr;
    float gz1 = (d11 * v0.z - d01 * v1.z) * denomr;
    float gz2 = (d00 * v1.z - d01 * v0.z) * denomr;

    static_assert(std::is_standard_layout<MeshSimplifier::Quadric>::value);
    memset(this, 0, sizeof(Quadric));

    this->w = w;

    auto attr_count = grads.size();
    for (auto i = 0u; i < attr_count; i++) {
        float a0 = attr0[i];
        float a1 = attr1[i];
        float a2 = attr2[i];

        float gx = gx1 * (a1 - a0) + gx2 * (a2 - a0);
        float gy = gy1 * (a1 - a0) + gy2 * (a2 - a0);
        float gz = gz1 * (a1 - a0) + gz2 * (a2 - a0);
        float gw = a0 - p0.x * gx - p0.y * gy - p0.z * gz;

        this->a00 += w * (gx * gx);
        this->a11 += w * (gy * gy);
        this->a22 += w * (gz * gz);

        this->a10 += w * (gy * gx);
        this->a20 += w * (gz * gx);
        this->a21 += w * (gz * gy);

        this->b0 += w * (gx * gw);
        this->b1 += w * (gy * gw);
        this->b2 += w * (gz * gw);

        this->c += w * (gw * gw);

        grads[i].gx = w * gx;
        grads[i].gy = w * gy;
        grads[i].gz = w * gz;
        grads[i].gw = w * gw;
    }
}

MeshSimplifier::QuadricGrad MeshSimplifier::QuadricGrad::operator+(const QuadricGrad &rhs) const {
    return {gx + rhs.gx, gy + rhs.gy, gz + rhs.gz, gw + rhs.gw};
}

MeshSimplifier::QuadricGrad &MeshSimplifier::QuadricGrad::operator+=(const QuadricGrad &rhs) {
    gx += rhs.gx;
    gy += rhs.gy;
    gz += rhs.gz;
    gw += rhs.gw;

    return *this;
}

MeshSimplifier::MeshSimplifier(MeshDataView view, bool inplace) :
    input_view(view),
    m_remain_num_vert(view.vertex_count),
    m_remain_num_tri(view.indices.size() / 3),
    inplace(inplace) {
    // build edge hash map
    if (inplace) {
        this->view = view;
    } else {
        m_indices          = std::vector<uint32_t>(view.indices.begin(), view.indices.end());
        this->view         = view;
        this->view.indices = std::span(m_indices);
    }

    const auto vertex_count = view.vertex_count;
    const auto attr_count   = view.attr_stride.size();

    remap.resize(vertex_count);
    wedge.resize(vertex_count);

    loop_back.resize(vertex_count, 0xFFFFFFFF);
    loop.resize(vertex_count, 0xFFFFFFFF);

    vertex_flags.resize(vertex_count, VertexKind::Locked);

    vertex_quadrics.resize(vertex_count);
    attr_quadrics.resize(vertex_count * attr_count);
    attr_quadric_grads.resize(vertex_count * attr_count);
}

void MeshSimplifier::set_material_ids(std::span<uint32_t> material_ids) {
    m_material_ids = material_ids;
}

void MeshSimplifier::lock_vertex(uint32_t idx) {
    // the locked vertex is recorded in the original mesh, this array will be used later during the
    // simplification
    locked_vertices.push_back(idx);
}

void MeshSimplifier::init_adjacency() {
    auto &offset = edge_adj.offset;
    auto &edge   = edge_adj.edge;

    offset.resize(input_view.vertex_count + 1, 0U);
    edge.resize(input_view.indices.size());

    const auto index_count  = input_view.indices.size();
    const auto face_count   = index_count / 3U;
    const auto vertex_count = input_view.vertex_count;

    for (auto i = 0U; i < index_count; i++) {
        const auto &idx = input_view.indices[i];
        assert(idx < vertex_count);
        offset[idx] += 1;
    }

    // convert from count to offset
    std::partial_sum(offset.begin(), offset.end(), offset.begin());
    std::rotate(offset.begin(), offset.end() - 1, offset.end());
    offset[0] = 0;

    assert(offset.back() == index_count);

    for (auto i = 0U; i < face_count; i++) {
        auto a = input_view.indices[i * 3 + 0];
        auto b = input_view.indices[i * 3 + 1];
        auto c = input_view.indices[i * 3 + 2];

        edge[offset[a]] = {b, c};
        offset[a]++;

        edge[offset[b]] = {c, a};
        offset[b]++;

        edge[offset[c]] = {a, b};
        offset[c]++;
    }

    std::rotate(offset.begin(), offset.end() - 1, offset.end());
    offset[0] = 0;
    assert(offset.back() == index_count);
}

void MeshSimplifier::update_adjacency() {
    auto &offset = edge_adj.offset;
    auto &edge   = edge_adj.edge;

    offset.resize(input_view.vertex_count + 1, 0U);
    edge.resize(input_view.indices.size());

    const auto index_count  = input_view.indices.size();
    const auto face_count   = index_count / 3U;
    const auto vertex_count = input_view.vertex_count;

    for (auto i = 0U; i < index_count; i++) {
        auto idx = input_view.indices[i];
        idx      = remap[idx];

        assert(idx < vertex_count);

        offset[idx] += 1;
    }

    // convert from count to offset
    std::partial_sum(offset.begin(), offset.end(), offset.begin());
    std::rotate(offset.begin(), offset.end() - 1, offset.end());
    offset[0] = 0;

    assert(offset.back() == index_count);

    for (auto i = 0U; i < face_count; i++) {
        auto a = input_view.indices[i * 3 + 0];
        auto b = input_view.indices[i * 3 + 1];
        auto c = input_view.indices[i * 3 + 2];

        a = remap[a];
        b = remap[b];
        c = remap[c];

        edge[offset[a]] = {b, c};
        offset[a]++;

        edge[offset[b]] = {c, a};
        offset[b]++;

        edge[offset[c]] = {a, b};
        offset[c]++;
    }

    std::rotate(offset.begin(), offset.end() - 1, offset.end());
    offset[0] = 0;
    assert(offset.back() == index_count);
}

void MeshSimplifier::remap_vertices() {
    std::iota(remap.begin(), remap.end(), 0U);
    std::iota(wedge.begin(), wedge.end(), 0U);

    // ! there exists problems, it there exists duplicate vertices, the classification will be wrong
    // ! so we need to remove the duplicate vertices first, I mean, better during the loading stage
    // ! here, duplication means the vertex data are totally the same, including the attributes
    const auto position_getter
        = [&](uint32_t idx) -> const Position & { return input_view.get_position_indexed(idx); };

    PositionMap<uint32_t, decltype(position_getter)> position_map(position_getter,
                                                                  input_view.vertex_count);

    for (auto i = 0U; i < input_view.vertex_count; ++i) {
        // if point already exists
        if (position_map.count(i) == 0) {
            remap[i] = i;
            position_map.emplace(i, i);
        } else {
            remap[i] = position_map.at(i);
        }
    }

    // build the wedge map, wedge means the vertex with the same position, the data struct is a
    // linked cycle list, so you can traverse the wedge by index, wedge[wedge[i]]
    for (auto i = 0U; i < input_view.vertex_count; ++i) {
        if (remap[i] != i) {
            auto r   = remap[i];
            wedge[i] = wedge[r];
            wedge[r] = i;
        }
    }
}

bool MeshSimplifier::has_edge(uint32_t source, uint32_t target) {
    auto start = edge_adj.offset[source];
    auto end   = edge_adj.offset[source + 1];

    const auto &edges = edge_adj.edge;

    return std::any_of(edges.cbegin() + start, edges.cbegin() + end,
                       [&](const auto &edge) { return edge.next == target; });
}

// ! compared with unreal, the classification is not well behaved, change to the unreal version
void MeshSimplifier::classify_vertices() {
    // only valid for border/seam
    std::fill(loop_back.begin(), loop_back.end(), 0xFFFFFFFF);
    std::fill(loop.begin(), loop.end(), 0xFFFFFFFF);

    const auto &vertex_count = input_view.vertex_count;

    for (auto i = 0U; i < vertex_count; ++i) {
        auto vertex = i;
        auto offset = edge_adj.offset[vertex];
        auto count  = edge_adj.offset[i + 1] - offset;

        for (auto j = 0; i < count; j++) {
            auto edge = edge_adj.edge[offset + j];

            auto target = edge.next;
            if (target == vertex) {
                // degenerate triangle, since one edge of this triangle hash the same source and
                // target, and the self edge are bidirectional.
                loop[vertex] = loop_back[vertex] = vertex;
            } else if (!has_edge(target, vertex)) { // not shared edge
                // dont overwrite the degenerate triangle information
                // 1. border
                // 2. degenerate
                loop[vertex]      = (loop[vertex] == 0xFFFFFFFF) ? target : vertex;
                loop_back[vertex] = (loop_back[vertex] == 0xFFFFFFFF) ? target : vertex;
            }
        }
    }

    // classify
    // TODO: Think about this part again
    for (auto i = 0U; i < vertex_count; ++i) {
        if (remap[i] == i) {
            if (wedge[i] == i) {
                // no attr seam
                auto openi = loop_back[i];
                auto openo = loop[i];

                if (openi == 0xFFFFFFFF && openo == 0xFFFFFFFF) {
                    // manifold
                    vertex_flags[i] = VertexKind::Manifold;
                } else if (openi != i && openo != i) {
                    // border
                    vertex_flags[i] = VertexKind::Border;
                } else {
                    // TODO: I guess such triangles should also be removed?
                    vertex_flags[i] = VertexKind::Locked;
                }
            } else {
                // wedge is like a single linked circle list
                auto w = wedge[i];
            }
        } else {
            assert(remap[i] < i);
            vertex_flags[i] = vertex_flags[remap[i]];
        }
    }
}

void MeshSimplifier::fix_vertex() {
    // fix vertex strange values
}

void MeshSimplifier::update_quadrics() {
    memset(vertex_quadrics.data(), 0, sizeof(Quadric) * vertex_quadrics.size());
    if (attr_quadrics.size() > 0) {
        memset(attr_quadrics.data(), 0, sizeof(Quadric) * attr_quadrics.size());
        memset(attr_quadric_grads.data(), 0, sizeof(QuadricGrad) * attr_quadric_grads.size());
    }

    const auto vertex_count = input_view.vertex_count;
    const auto index_count  = input_view.indices.size();
    const auto attr_count   = input_view.attr_stride.size();
    // face quadrics
    for (auto i = 0U; i < index_count; i += 3) {
        auto a = input_view.indices[i * 3 + 0];
        auto b = input_view.indices[i * 3 + 1];
        auto c = input_view.indices[i * 3 + 2];

        Quadric q;
        q.fromTriangle(input_view.get_position(a), input_view.get_position(b),
                       input_view.get_position(c));

        vertex_quadrics[remap[a]] += q;
        vertex_quadrics[remap[b]] += q;
        vertex_quadrics[remap[c]] += q;
    }
    // edge quadrics
    for (auto i = 0U; i < index_count; i += 3) {
        static constexpr uint32_t next[] = {1, 2, 0, 1};
        for (auto e : std::views::iota(0U, 3U)) {
            auto i0 = input_view.indices[i + e];
            auto i1 = input_view.indices[i + next[e]];

            auto k0 = vertex_flags[i0];
            auto k1 = vertex_flags[i1];

            if (k0 != VertexKind::Border && k0 != VertexKind::Seam && k1 != VertexKind::Border
                && k1 != VertexKind::Seam) {
                continue;
            }

            if ((k0 == VertexKind::Border || k0 == VertexKind::Seam) && loop[i0] != i1) {
                continue;
            }

            if ((k1 == VertexKind::Border || k1 == VertexKind::Seam) && loop_back[i1] != i0) {
                continue;
            }

            if (kHasOpposite[k0][k1] && remap[i1] > remap[i0]) {
                continue;
            }

            auto i2 = input_view.indices[i + next[e + 1]];

            // we try hard to maintain border edge geometry; seam edges can move more freely
            // due to topological restrictions on collapses, seam quadrics slightly improves
            // collapse structure but aren't critical
            const float kEdgeWeightSeam   = 1.f;
            const float kEdgeWeightBorder = 10.f;

            float edgeWeight = (k0 == VertexKind::Border || k1 == VertexKind::Border)
                                   ? kEdgeWeightBorder
                                   : kEdgeWeightSeam;

            Quadric q;
            q.fromTriangleEdge(input_view.get_position(i0), input_view.get_position(i1),
                               input_view.get_position(i2), edgeWeight);
            vertex_quadrics[remap[i0]] += q;
            vertex_quadrics[remap[i1]] += q;
        }
    }
    // build attr quadrics
    if (attr_count > 0) {
        for (auto i = 0U; i < index_count; i += 3) {
            auto a = input_view.indices[i * 3 + 0];
            auto b = input_view.indices[i * 3 + 1];
            auto c = input_view.indices[i * 3 + 2];

            Quadric     q;
            QuadricGrad g[kMaxAttributes];

            q.fromAttributes(g, input_view.get_position(a), input_view.get_position(b),
                             input_view.get_position(c), input_view.get_attr(a),
                             input_view.get_attr(b), input_view.get_attr(c));
            attr_quadrics[remap[a]] += q;
            attr_quadrics[remap[b]] += q;
            attr_quadrics[remap[c]] += q;

            for (auto j = 0U; j < attr_count; ++j) {
                attr_quadric_grads[remap[a] * attr_count + j] += g[j];
                attr_quadric_grads[remap[b] * attr_count + j] += g[j];
                attr_quadric_grads[remap[c] * attr_count + j] += g[j];
            }
        }
    }
}

void MeshSimplifier::init_collapse() {
    auto dual_count = 0U;
    for (auto i : std::views::iota(0U, input_view.vertex_count)) {
        auto k = vertex_flags[i];
        auto e = edge_adj.offset[i + 1] - edge_adj.offset[i];

        if (k == VertexKind::Manifold || k == VertexKind::Seam) {
            dual_count += e;
        }
    }

    const auto collapse_size = input_view.indices.size() - dual_count / 2 + 3;

    collapse_data.resize(collapse_size);
}

uint32_t MeshSimplifier::update_collapse() {
    size_t collapse_count = 0;

    uint32_t index_count = input_view.indices.size();

    for (uint32_t i = 0; i < index_count; i += 3) {
    }

    // TODO impl

    return 0;
}

void MeshSimplifier::simplify(size_t target_num_vert, size_t target_num_tri, float target_error) {
    // lock vertices, wedge is linked list
    for (auto i : locked_vertices) {
        vertex_flags[i] = VertexKind::Locked;
        uint32_t next   = wedge[i];
        while (next != i) {
            vertex_flags[next] = VertexKind::Locked;
            next               = wedge[next];
        }
    }

    auto dual_count = 0U;
    for (auto i : std::views::iota(0U, input_view.vertex_count)) {
        auto k = vertex_flags[i];
        auto e = edge_adj.offset[i + 1] - edge_adj.offset[i];

        if (k == VertexKind::Manifold || k == VertexKind::Seam) {
            dual_count += e;
        }
    }

    const auto collapse_size = input_view.indices.size() - dual_count / 2 + 3;

    std::vector<Collapse> collapse_data(collapse_size);
    std::vector<uint32_t> collapse_remap(input_view.vertex_count, 0xFFFFFFFF);
    std::vector<uint32_t> collapse_order(input_view.vertex_count, 0xFFFFFFFF);
    std::vector<uint8_t>  collapse_flags(input_view.vertex_count, 0);

    size_t result_count       = input_view.indices.size();
    size_t target_index_count = target_num_vert * 3;

    float result_error = 0.f;

    float error_threshold = target_error * target_error;

    while (result_count > target_index_count) {
        update_adjacency();
        // size_t collapse_count = 0;
    }
}

void MeshSimplifier::dump_obj(std::string_view filename) {
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