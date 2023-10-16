#include "utility.hpp"
#include "builder.hpp"
#include <algorithm>
#include <boost/graph/properties.hpp>
#include <stdint.h>

#if ENABLE_TEST
#    include <gtest/gtest.h>
#endif

#if ENABLE_TEST

TEST(Triangle, cycle) {
    EXPECT_EQ(triangle_cycle(0), 1);
    EXPECT_EQ(triangle_cycle(1), 2);
    EXPECT_EQ(triangle_cycle(2), 0);
}

#endif

#if ENABLE_TEST

TEST(IOTA, enumerate) {
    std::vector<uint32_t> a(10);
    std::iota(a.begin(), a.end(), 0);
    std::vector<uint32_t> b{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_EQ(a, b);
}

#endif

#if ENABLE_TEST

TEST(algorithm, hashmap_adapter) {
    std::vector<Position> positions{
        {0, 0, 0},
        {1, 1, 1},
        {2, 2, 2},
    };

    std::vector<uint32_t> indices{0, 2, 2, 1, 2, 1};

    auto positions_getter
        = [&](uint32_t idx) -> const Position & { return positions[indices[idx]]; };

    struct PositionGetter {
        PositionGetter(std::vector<Position> &vert, std::vector<uint32_t> &indices) :
            positions(vert), indices(indices) {
        }

        const Position &operator()(uint32_t idx) const {
            return positions[indices[idx]];
        }

        const std::vector<Position> &positions;
        const std::vector<uint32_t> &indices;
    };

    PositionMultiMap<uint32_t, PositionGetter> map(PositionGetter(positions, indices), 100);

    for (auto i = 0U; i < indices.size(); ++i) {
        map.map.emplace(i, indices[i]);
    }

    for (auto i = 0U; i < indices.size(); ++i) {
        EXPECT_EQ(map.map.find(i)->second, indices[i]);
    }

    auto range = map.map.equal_range(2);
    EXPECT_EQ(std::distance(range.first, range.second), 3);
}

#endif

Bounds get_bound(const MeshDataView &view) {
    Bounds bound{};

    for (auto i = 0U; i < view.indices.size(); ++i) {
        const auto &pos = view.get_vertex(view.indices[i]);
        bound           = extend(bound, pos);
    }

    return bound;
}

SphereBounds sphere_from_points(std::span<const Position> points) {
    SphereBounds bound;

    uint32_t min_idx[3] = {0, 0, 0};
    uint32_t max_idx[3] = {0, 0, 0};

    for (uint32_t i = 0; i < points.size(); ++i) {
        for (uint32_t j = 0; j < 3; ++j) {
            if (points[i].data[j] < points[min_idx[j]].data[j]) {
                min_idx[j] = i;
            }
            if (points[i].data[j] > points[max_idx[j]].data[j]) {
                max_idx[j] = i;
            }
        }
    }

    float    largestDist = 0.0f;
    uint32_t largestAxis = 0;

    for (uint32_t i = 0; i < 3; ++i) {
        float distSqr = length((points[max_idx[i]] - points[min_idx[i]]));
        if (distSqr > largestDist) {
            largestDist = distSqr;
            largestAxis = i;
        }
    }

    Position center = (points[min_idx[largestAxis]] + points[max_idx[largestAxis]]) * 0.5f;
    float    radius = 0.0f;

    for (uint32_t i = 0; i < points.size(); i++) {
        float dist = length((points[i] - center));
        if (dist > radius) {
            float t = 0.5 + 0.5f * (radius / dist);
            center  = lerp(points[i], center, t);
            radius  = 0.5f * (radius + dist);
        }
    }

    bound.radius = radius;
    for (uint32_t i = 0; i < 3; ++i) {
        bound.center[i] = (center.data[i] + center.data[i]) * 0.5f;
    }

    return bound;
}

Position &get_sphere_center(SphereBounds &bound) {
    return *reinterpret_cast<Position *>(&bound.center);
}

const Position &get_sphere_center(const SphereBounds &bound) {
    return *reinterpret_cast<const Position *>(&bound.center);
}

void sphere_add(SphereBounds &bound, const SphereBounds &other) {
    float dist = length(get_sphere_center(bound) - get_sphere_center(other));
    if (dist + other.radius <= bound.radius) {
        return;
    } else if (dist + bound.radius <= other.radius) {
        bound = other;
        return;
    } else {
        float    t      = 0.5 + 0.5f * (bound.radius - other.radius) / dist;
        Position center = lerp(get_sphere_center(bound), get_sphere_center(other), t);
        for (uint32_t i = 0; i < 3; ++i) {
            bound.center[i] = center.data[i];
        }
        bound.radius = 0.5f * (dist + bound.radius + other.radius);
    }
}

SphereBounds sphere_from_balls(std::span<const SphereBounds> balls) {
    uint32_t min_idx[3] = {0, 0, 0};
    uint32_t max_idx[3] = {0, 0, 0};

    for (uint32_t i = 0; i < balls.size(); ++i) {
        for (uint32_t j = 0; j < 3; ++j) {
            if (balls[i].center[j] - balls[i].radius
                < balls[min_idx[j]].center[j] - balls[min_idx[j]].radius) {
                min_idx[j] = i;
            }
            if (balls[i].center[j] + balls[i].radius
                > balls[max_idx[j]].center[j] + balls[max_idx[j]].radius) {
                max_idx[j] = i;
            }
        }
    }

    float    largestDist = 0.0f;
    uint32_t largestAxis = 0;
    for (uint32_t i = 0; i < 3; ++i) {
        const auto &ball_min = balls[min_idx[i]];
        const auto &ball_max = balls[max_idx[i]];
        float       dist     = length(get_sphere_center(ball_max) - get_sphere_center(ball_min))
                     + ball_min.radius + ball_max.radius;

        if (dist > largestDist) {
            largestDist = dist;
            largestAxis = i;
        }
    }

    Position center = (get_sphere_center(balls[min_idx[largestAxis]])
                       + get_sphere_center(balls[max_idx[largestAxis]]))
                      * 0.5f;
    float radius = balls[min_idx[largestAxis]].radius
                   + length(get_sphere_center(balls[min_idx[largestAxis]]) - center);

    SphereBounds bound;
    for (uint32_t i = 0; i < 3; ++i) {
        bound.center[i] = center.data[i];
    }
    bound.radius = radius;

    for (uint32_t i = 0; i < balls.size(); i++) {
        sphere_add(bound, balls[i]);
    }
}

SphereBounds get_sphere_bound(const MeshDataView &view) {
    std::vector<Position> positions(view.indices.size());
    for (auto i = 0U; i < view.indices.size(); ++i) {
        positions[i] = view.get_position(view.indices[i]);
    }

    return sphere_from_points(positions);
}