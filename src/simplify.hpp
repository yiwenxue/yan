#pragma once

#include <cstdint>
#include <vector>

#include "hasher.hpp"
#include "simplify.hpp"
#include "utility.hpp"

#include "meshoptimizer.h"

class MeshSimplify {
public:
    MeshSimplify(MeshDataView view, bool inplace);

    float simplify(size_t target_num_tri, float target_error);

    void lock_position(uint32_t i);

    void dump_obj(std::string_view filename);

private:
    bool m_inplace;

    MeshDataView input_view;
    MeshDataView view;

    static meshopt_Allocator m_allocator;

    std::vector<uint32_t> m_indices;
    std::vector<uint32_t> locked_positions;

    void lock_positions();
};