#pragma once

#include <cstdint>
#include <initializer_list>
#include <type_traits>

static inline uint32_t murmur32_finalize(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return h;
}

static inline uint32_t murmur32(std::initializer_list<uint32_t> list) {
    uint32_t h = 0;
    for (auto e : list) {
        e *= 0xcc9e2d51;
        e = (e << 15) | (e >> 17);
        e *= 0x1b873593;

        h ^= e;
        h = (h << 13) | (h >> 19);
        h = h * 5 + 0xe6546b64;
    }
    return murmur32_finalize(h);
}

// TODO check validation
static inline uint32_t murmur64_finalize(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;

    return h;
}

static inline uint64_t murmur64(std::initializer_list<uint64_t> list) {
    uint64_t h = 0;
    for (auto e : list) {
        e *= 0xff51afd7ed558ccd;
        e = (e << 33) | (e >> 31);
        e *= 0xc4ceb9fe1a85ec53;

        h ^= e;
        h = (h << 31) | (h >> 33);
        h = h * 5 + 0xe6546b64;
    }
    return murmur64_finalize(h);
}