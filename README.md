# Cluster

Another implementation of nanite like mesh seamless LOD system.

## Features

## Dependencies

1. vcpkg
2. cmake, ninja, clang++
3. library
   1. boost graph
   2. metis
   3. c++ 20 (adapt to 17)
   4. coroutine (unifex)

## Usage

## License

## Credits

## TODO     

- [ ] Memory management, limit the memory usage, or the memory usage will be very large.
  - [x] wrap hash adapter, equality adapter, to reduce the hashmap memory usage.
  - [ ] use pmr to reduce the memory usage.
- [x] Add graph coloring algorithm to optimize the visualizer.
- [ ] Adaptive remesh to improve the cluster quality. not that difficult.
  - [x] verified that the adaptive helps to improve the cluster quality.
  - [ ] implement a simple adaptive remesh algorithm. // should we?
- [ ] concurrent cluster building to save time
- [x] mesh simplifier
  - [x] lock edge
- [x] cluster group generation
  - [x] group simplify
  - [x] parent cluster generation
  - [x] cluster DAG
- [ ] serialization
- [ ] fix cracks between sub-meshes
- [ ] cluster vertex quantization
  - [ ] position is easy, what about others?

## Changelog

## Links

## Contact