#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric> // for std::iota
#include <optional>
#include <span>
#include <stack>
#include <string>
#include <vector>

template <std::floating_point T>
struct vec3 {
    union {
        T x, y, z;
        T data[3];
    };
};

template <std::floating_point T>
vec3<T> operator-(const vec3<T> &lhs, const vec3<T> &rhs) {
    return {.data = {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z}};
}

template <std::floating_point T>
vec3<T> operator+(const vec3<T> &lhs, const vec3<T> &rhs) {
    return {.data = {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z}};
}

template <std::floating_point T>
struct bbox {
    vec3<T> min{std::numeric_limits<T>::max()};
    vec3<T> max{std::numeric_limits<T>::lowest()};
};

template <std::floating_point T>
bbox<T> merge(const bbox<T> &a, const bbox<T> &b) {
    return {.min = {.data = {std::min(a.min.x, b.min.x), std::min(a.min.y, b.min.y),
                             std::min(a.min.z, b.min.z)}},
            .max = {.data = {std::max(a.max.x, b.max.x), std::max(a.max.y, b.max.y),
                             std::max(a.max.z, b.max.z)}}};
}

template <std::floating_point T>
float bbox_half_area(const bbox<T> &box) {
    const auto dx = box.max.x - box.min.x;
    const auto dy = box.max.y - box.min.y;
    const auto dz = box.max.z - box.min.z;
    return dx * dy + dy * dz + dz * dx;
}

template <std::floating_point T>
vec3<T> bbox_center(const bbox<T> &box) {
    return {.data = {(box.min.x + box.max.x) * 0.5f, (box.min.y + box.max.y) * 0.5f,
                     (box.min.z + box.max.z) * 0.5f}};
}

/**
 * Markdown
 *
 * general purpose spatial acceleration data structure
 * 1. object
 *    x object stored in leaf nodes
 *    x object ref stored in internal nodes
 *    * object index stored in internal nodes
 * 2. bounding box
 *    x bounding box stored in leaf nodes
 *    * bounding box stored in tree and parallel to nodes array
 * 3. tree
 * 4. constructor
 *   * constructor of different strategies
 * 5. traversal
 *   * traversal of different strategies
 *   * iterative traversal
 * 7. update
 *
 */

// k dimensional tree
template <typename Data, size_t Dimension, std::unsigned_integral Index,
          std::floating_point Precision>
struct kdtree {
    using index_t    = Index;
    using float_t    = Precision;
    using data_t     = Data;
    using dim        = std::integral_constant<size_t, Dimension>;
    using position_t = std::array<float_t, Dimension>;

    struct node_t {
        index_t left;
        index_t right;
        index_t split;
        index_t object;
    };

    std::vector<node_t>     m_nodes;
    std::vector<data_t *>   m_objects;
    std::vector<position_t> m_positions;
};

template <typename KdTree>
struct kdtree_traits {
    using data_t     = typename KdTree::data_t;
    using index_t    = typename KdTree::index_t;
    using float_t    = typename KdTree::float_t;
    using dim        = typename KdTree::dim;
    using position_t = typename KdTree::position_t;
    using node_t     = typename KdTree::node_t;
};

template <typename KdTree>
struct default_kdtree_constructor {
    default_kdtree_constructor() = default;
};

template <typename Data, std::unsigned_integral Index, std::integral Dimension = uint32_t,
          std::floating_point Precision = float>
struct octree {
    using data_t  = Data;
    using index_t = Index;
    using dim_t   = Dimension;
    using float_t = Precision;
    struct node_t {
        index_t parent = -1;     // index to parent
        index_t object = -1;     // index to object
        index_t children[8]{-1}; // index to children
    };

    std::vector<node_t>   m_nodes;
    std::vector<data_t *> m_object;
    index_t               m_root = 0xFFFFFFFF;

    dim_t         m_dim = 0;
    vec3<float_t> m_min{0};
    vec3<float_t> m_max{0};

    struct iterator_t {};

    octree() = default;
    octree(dim_t dim, vec3<float_t> min, vec3<float_t> max) : m_dim(dim), m_min(min), m_max(max) {
    }
};

template <typename Octree>
struct octree_traits {
    using data_t  = typename Octree::data_t;
    using index_t = typename Octree::index_t;
    using dim_t   = typename Octree::dim_t;
    using float_t = typename Octree::float_t;
    using node_t  = typename Octree::node_t;
};

template <typename Octree>
struct default_octree_constructor {
    default_octree_constructor() {
    }

    Octree build(std::span<typename Octree::data_t *> objects) {
        Octree tree{};
        return std::move(tree);
    }
};

template <typename Octree>
struct octree_traversal {};

template <typename Data, std::unsigned_integral Index = uint32_t,
          std::floating_point Precision = float>
struct bvh {
    using index_t = Index;
    using data_t  = Data;
    using float_t = Precision;
    struct node_t {
        index_t parent = -1; // index to parent
        index_t left   = -1; // index to left child
        index_t right  = -1; // index to right child
        index_t object = -1; // index to object
    };

    std::vector<data_t *>      m_object;
    std::vector<node_t>        m_nodes;
    std::vector<bbox<float_t>> m_boxes;

    index_t num_objects;
    index_t num_nodes;

    struct iterator_t {};
};

template <typename BVH>
struct bvh_traits {
    using data_t  = typename BVH::data_t;
    using index_t = typename BVH::index_t;
    using float_t = typename BVH::float_t;
    using node_t  = typename BVH::node_t;
};

template <typename Object>
struct bbox_getter {};

/**
 * @brief default bvh constructor, build strategy is top-down, recursive
 * user can override the make_split function to change the split strategy
 *
 * @tparam BVH is the bvh type
 * @tparam BboxGetter is the bounding box getter
 */
template <typename BVH, std::invocable<const typename bvh_traits<BVH>::data_t &> BboxGetter =
                            bbox_getter<typename bvh_traits<BVH>::data_t>>
struct default_bvh_constructor {
    using data_t  = typename bvh_traits<BVH>::data_t;
    using index_t = typename bvh_traits<BVH>::index_t;
    using float_t = typename bvh_traits<BVH>::float_t;
    using node_t  = typename bvh_traits<BVH>::node_t;

    using cost_function = std::function<float_t(index_t, index_t)>;

    default_bvh_constructor() {
    }

    // return the split index, 0 means no split
    // it is a virtual function, so it can be overrided, e.g. median split, sah split
    virtual index_t make_split(BVH &tree, index_t start, index_t end) {
        // just a simple median split
        if (end - start <= 1) {
            return 0;
        }

        // compute the bounding box
        bbox box = compute_bbox(start, end);

        // find the longest axis
        float_t  max_length = 0;
        uint32_t max_axis   = 0;
        for (uint32_t axis = 0; axis < 3; axis++) {
            float_t length = box.max.data[axis] - box.min.data[axis];
            if (length > max_length) {
                max_length = length;
                max_axis   = axis;
            }
        }

        // sort the objects along the longest axis
        std::sort(tree.m_object.begin() + start, tree.m_object.begin() + end,
                  [&](data_t *a, data_t *b) {
                      auto box_a = bbox_center(BboxGetter()(*a));
                      auto box_b = bbox_center(BboxGetter()(*b));
                      return box_a.data[max_axis] < box_b.data[max_axis];
                      return true;
                  });

        // find the split position
        index_t split = start + (end - start) / 2;

        // if the split position is the same as start or end, then no split
        if (split == start || split == end) {
            return 0;
        }

        return split;
    }

    BVH build(std::span<data_t *> objects) {
        BVH tree{};

        tree.m_num_nodes = objects.size() * 2 - 1;
        tree.m_nodes.reserve(tree.m_num_nodes);
        tree.m_boxes.reserve(tree.m_num_nodes);

        tree.m_num_objects = objects.size();
        tree.m_object.reserve(tree.m_num_objects);
        std::copy(objects.begin(), objects.end(), std::back_inserter(tree.m_object));

        // stack based construction
        struct item {
            index_t start;
            index_t end;
            index_t node;
        };

        std::stack<item> stack;
        stack.push({0, tree.m_num_objects, 0});

        while (!stack.empty()) {
            auto [start, end, node] = stack.top();
            stack.pop();

            auto split = make_split(tree, start, end);
            if (split != 0) {
                // internal node
                auto left_box  = compute_bbox(start, split);
                auto right_box = compute_bbox(split, end);

                auto left_part  = std::make_pair(start, split);
                auto right_part = std::make_pair(split, end);

                // For "any-hit" queries, the left child is chosen first, so we make sure that
                // it is the child with the largest area, as it is more likely to contain an
                // an occluder.
                if (bbox_half_area(left_box) < bbox_half_area(right_box)) {
                    std::swap(left_box, right_box);
                    std::swap(left_part, right_part);
                }

                tree.m_boxes.push_back(left_box);
                tree.m_boxes.push_back(right_box);

                tree.m_nodes[node].left  = tree.m_nodes.size();
                tree.m_nodes[node].right = tree.m_nodes.size() + 1;

                tree.m_nodes.push_back({node, -1, -1, -1});
                tree.m_nodes.push_back({node, -1, -1, -1});

                auto item1 = item{left_part.first, left_part.second, tree.m_nodes[node].left};
                auto item2 = item{right_part.first, right_part.second, tree.m_nodes[node].right};

                // Process the largest child item first, in order to minimize the stack size.
                if (item1.end - item1.start < item2.end - item2.start) {
                    std::swap(item1, item2);
                }
                stack.push(std::move(item1));
                stack.push(std::move(item2));
            } else {
                // leaf node
                tree.m_nodes[node].object = start;
            }
        }

        return std::move(tree);
    }

    // TODO error, should use aabb from objects
    bbox<float_t> compute_bbox(size_t start, size_t end) const {
        bbox<float_t> box{};
        for (auto i = start; i < end; i++) {
            box = merge(box, bboxes[i]);
        }
        return box;
    }

    std::vector<bbox<float_t>> bboxes;
};

template <typename BVH>
struct bvh_traversal {};

// test
template <std::floating_point T>
struct sphere {
    vec3<T> center;
    T       radius;
};

template <std::floating_point T>
struct bbox_getter<sphere<T>> {
    bbox<T> operator()(const sphere<T> &s) {
        return bbox<T>{s.center - vec3<T>{s.radius}, s.center + vec3<T>{s.radius}};
    }
};

using sphere_bvh = bvh<sphere<float>, uint32_t>;

default_bvh_constructor<sphere_bvh> constructor;
