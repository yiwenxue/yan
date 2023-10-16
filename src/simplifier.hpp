#pragma once

#include <cstdint>
#include <vector>

#include "hasher.hpp"
#include "simplify.hpp"
#include "utility.hpp"

// all attributes must be float
// static mesh simplifier class
// must have
// 1. position     (3)
// 2. normal       (3)
// optional
// 3. tangent      (3)
// 4. texcoord[n]  (2*n)
// 4+n. color      (4)
class MeshSimplifier {
public:
    MeshSimplifier(MeshDataView view, bool inplace);

    // the simplify algorithm should support multiple sub-meshes
    void set_material_ids(std::span<uint32_t> material_ids);

    void simplify(size_t target_num_vert, size_t target_num_tri, float target_error);

    void compact();

    void lock_vertex(uint32_t idx);

    void dump_obj(std::string_view filename);

private:
    enum VertexKind {
        Manifold = 0, // not attr seam, not border
        Border,       // has exactly two open half-edges
        Seam,         // has exactly two attr seam edges
        Complex,
        Locked, // none of above or locked by user

        Count,
    };

    static constexpr uint32_t  VertexKindCount = VertexKind::Count;
    static const unsigned char kCanCollapse[VertexKindCount][VertexKindCount];
    static const unsigned char kHasOpposite[VertexKindCount][VertexKindCount];

    struct EdgeAdjacency {
        struct Edge {
            union {
                struct {
                    uint32_t next;
                    uint32_t prev;
                };
                uint64_t data;
            };
        };

        std::vector<uint32_t> offset;
        std::vector<Edge>     edge;
    };

    static constexpr uint32_t kMaxAttributes = 16;

    struct QuadricGrad {
        float gx, gy, gz, gw;

        QuadricGrad  operator+(const QuadricGrad &rhs) const;
        QuadricGrad &operator+=(const QuadricGrad &rhs);
    };

    struct Quadric {
        float a00, a11, a22;
        float a10, a20, a21;
        float b0, b1, b2, c;
        float w;

        float operator()(const Position &p) const;

        Quadric  operator+(const Quadric &rhs) const;
        Quadric &operator+=(const Quadric &rhs);

        void fromPoint(const Position &p, float weight = 1.0F);
        void fromPlane(const Position &n, float d, float weight = 1.0F);
        void fromTriangle(const Position &p0, const Position &p1, const Position &p2,
                          float weight = 1.0F);
        void fromTriangleEdge(const Position &p0, const Position &p1, const Position &p2,
                              float weight = 1.0F);
        void fromAttributes(std::span<QuadricGrad> grads, const Position &p0, const Position &p1,
                            const Position &p2, const float *attr0, const float *attr1,
                            const float *attr2);
    };

    struct Collapse {
        uint32_t v0;
        uint32_t v1;
        union {
            uint32_t bidi;
            uint32_t errorui;
            float    error;
        };
    };

    MeshDataView input_view;
    MeshDataView view;

    bool inplace = false;

    std::vector<float>    m_vertices;
    std::vector<uint32_t> m_indices;
    std::vector<uint32_t> locked_vertices;

    std::span<uint32_t> m_material_ids; // same size as triangle count

    size_t m_num_vert{0};
    size_t m_num_index{0};
    size_t m_num_tri{0};

    size_t m_remain_num_vert;
    size_t m_remain_num_tri;

    std::vector<uint8_t> vertex_flags;

    std::vector<uint32_t> remap;
    std::vector<uint32_t> wedge;
    std::vector<uint32_t> loop_back;
    std::vector<uint32_t> loop;

    EdgeAdjacency edge_adj;

    std::vector<Quadric>     vertex_quadrics;
    std::vector<Quadric>     attr_quadrics;
    std::vector<QuadricGrad> attr_quadric_grads;

    std::vector<Collapse> collapse_data;

    void remap_vertices();

    void classify_vertices();

    void init_adjacency();

    void update_adjacency();

    bool has_edge(uint32_t source, uint32_t target);

    void fix_vertex();

    void update_quadrics();

    void init_collapse();

    uint32_t update_collapse();
};