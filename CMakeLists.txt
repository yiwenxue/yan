cmake_minimum_required(VERSION 3.23.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

include(${CMAKE_SOURCE_DIR}/cmake/vcpkg_bootstrap.cmake)
x_vcpkg_bootstrap()
set(VCPKG_BOOTSTRAP_OPTIONS "-disableMetrics")
set(VCPKG_INSTALL_OPTIONS "--no-print-usage")
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file")
list(APPEND VCPKG_FEATURE_FLAGS manifests)

# detect platform
if (WIN32)
  set (VCPKG_TARGET_TRIPLET x64-windows)
elseif (UNIX)
  set (VCPKG_TARGET_TRIPLET x64-linux)
endif()

project(Cluster LANGUAGES CXX)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(unifex CONFIG REQUIRED)
add_library(std::coroutines INTERFACE IMPORTED)
find_package(imgui CONFIG REQUIRED)
find_package(metis CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GTest REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

set (Boost_NO_WARN_NEW_VERSIONS 1)
find_package(Boost REQUIRED COMPONENTS graph program_options)

add_executable(builder
  tools/builder.cpp
  src/builder.cpp
  src/builder.hpp
  src/simplifier.cpp
  src/simplifier.hpp
  src/simplify.cpp
  src/simplify.hpp
  # src/meshopt_simplify.cpp
  # src/meshoptimizer.h
  src/utility.cpp
  src/utility.hpp
  src/partitioner.hpp
  src/partitioner.cpp
  src/bvh.cpp
  src/bvh.hpp
  src/cluster.cpp
  src/cluster.hpp
  src/cluster_group.cpp
  src/cluster_group.hpp
)

target_include_directories(builder
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(builder
  PRIVATE
    Boost::graph
    ${Boost_LIBRARIES}
    uuid
    unifex::unifex
    imgui::imgui
    metis
    glfw
)


add_executable(simplifier
  tools/simplifier.cpp
  src/builder.cpp
  src/builder.hpp
  src/simplifier.cpp
  src/simplifier.hpp
  src/simplify.cpp
  src/simplify.hpp
  src/utility.cpp
  src/utility.hpp
  src/partitioner.hpp
  src/partitioner.cpp
  src/bvh.cpp
  src/bvh.hpp
  src/cluster.cpp
  src/cluster.hpp
  src/cluster_group.cpp
  src/cluster_group.hpp
)

target_include_directories(simplifier
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(simplifier
  PRIVATE
    Boost::graph
    ${Boost_LIBRARIES}
    uuid
    unifex::unifex
    imgui::imgui
    metis
    glfw
)

enable_testing()

set (FILESYSTEM_ROOT "${CMAKE_SOURCE_DIR}")

configure_file(
  ${CMAKE_SOURCE_DIR}/config.h.in
  ${CMAKE_SOURCE_DIR}/visualizer/config.h
)

configure_file(
  ${CMAKE_SOURCE_DIR}/config.h.in
  ${CMAKE_SOURCE_DIR}/tools/config.h
)

file (GLOB_RECURSE VISUALIZER_SOURCES "visualizer/*.cpp" "visualizer/*.h")
file (GLOB_RECURSE VISUALIZER_GLSL_SOURCES "visualizer/shaders/*.vert" "visualizer/shaders/*.frag" "visualizer/shaders/*.comp")

add_executable(visualizer
  ${VISUALIZER_SOURCES}
  ${VISUALIZER_GLSL_SOURCES}
  src/builder.cpp
  src/builder.hpp
  src/simplifier.cpp
  src/simplifier.hpp
  src/simplify.cpp
  src/simplify.hpp
  src/utility.cpp
  src/utility.hpp
  src/partitioner.hpp
  src/partitioner.cpp
  src/bvh.cpp
  src/bvh.hpp
  src/cluster.cpp
  src/cluster.hpp
  src/cluster_group.cpp
  src/cluster_group.hpp
)

target_include_directories(visualizer
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/visualizer
)

find_package(Vulkan REQUIRED)

target_link_libraries(visualizer
  PRIVATE
    Boost::graph
    Boost::program_options
    uuid
    unifex::unifex
    imgui::imgui
    metis
    glfw
    Vulkan::Vulkan
    GPUOpen::VulkanMemoryAllocator
)

if (WIN32)
  # if build type is Debug, define _ITERATOR_DEBUG_LEVEL=1
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(visualizer
      PRIVATE
        -D_ITERATOR_DEBUG_LEVEL=1
    )
  endif()
endif()

if(UNIX)
  if(DEFINED ENV{VULKAN_SDK})
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/bin/glslangValidator")
  else()
    set(GLSL_VALIDATOR "glslangValidator")
  endif()
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
  set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin/glslangValidator.exe")
else()
  set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin32/glslangValidator.exe")
endif()

function (compile_shaders)
  set(options)
  set(oneValueArgs TARGET)

  cmake_parse_arguments(compile_shaders "${options}" "${oneValueArgs}" "" ${ARGN})
  get_target_property(target_sources ${compile_shaders_TARGET} SOURCES)
  get_target_property(target_source_dirs ${compile_shaders_TARGET} SOURCE_DIR)

  set(shader_sources)
  foreach(source ${target_sources})
    get_filename_component(source_ext ${source} EXT)
    if (${source_ext} STREQUAL ".vert" OR ${source_ext} STREQUAL ".frag" OR ${source_ext} STREQUAL ".comp")
      # search if ${source} contains ${target_source_dirs}
      string(FIND ${source} ${target_source_dirs} source_dir_index)
      if (NOT ${source_dir_index} EQUAL 0)
        list(APPEND shader_sources ${target_source_dirs}/${source})
      else()
        list(APPEND shader_sources ${source})
      endif()
    endif()
  endforeach()

  # compile all shaders
  foreach(source ${shader_sources})
    get_filename_component(source_we ${source} NAME_WE)
    get_filename_component(source_ext ${source} EXT)
    set(output ${CMAKE_BINARY_DIR}/shaders/${source_we}${source_ext}.spv)
    add_custom_command(
      OUTPUT ${output}
      COMMAND ${GLSL_VALIDATOR} -V ${source} -o ${output}
      DEPENDS ${source}
      COMMENT "Compiling shader ${source}"
      VERBATIM
    )
    list(APPEND shader_outputs ${output})
  endforeach()

  # add custom target
  add_custom_target(
    ${compile_shaders_TARGET}_shaders
    DEPENDS ${shader_outputs}
  )

  # add dependency to the target
  add_dependencies(${compile_shaders_TARGET} ${compile_shaders_TARGET}_shaders)
endfunction()

compile_shaders(TARGET visualizer)