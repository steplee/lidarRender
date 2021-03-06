project(tdt)
cmake_minimum_required(VERSION 3.16)

find_package(glfw3 REQUIRED)
set(GL_LIBS GL GLEW glfw GLU)
include_directories(/usr/local/include/eigen3/)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "-g -std=c++17")

add_library(tdt
  SHARED
  tdt/gltf.cc
  tdt/gltf_entity.cc
  tdt/render_context.cc
  tdt/tdt.cc

  tdt/extra_entities.cc
  tdt/camera.cc
  tdt/controllable_camera.cc

  tdt/tdtiles/parse.cc
  tdt/tdtiles/tile.cc
  tdt/tdtiles/sight.cc

  tdt/window.cc
  tdt/math.cc
  )


add_executable(test_tdt
  test/test_tdt.cc)
add_executable(test_gltf
  test/test.cc)
add_executable(printInfo
  test/print_info.cc)
add_executable(test_serialize
  test/test_serialize.cc)

add_executable(test_sight
  test/test_sight.cc)

target_link_libraries(tdt ${GL_LIBS})
target_link_libraries(test_gltf tdt)
target_link_libraries(test_tdt tdt)
target_link_libraries(test_serialize tdt)
target_link_libraries(test_sight tdt)
target_link_libraries(printInfo tdt)
