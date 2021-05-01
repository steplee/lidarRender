project(tdt)
cmake_minimum_required(VERSION 3.16)

find_package(glfw3 REQUIRED)
set(GL_LIBS GL GLEW glfw GLU)
include_directories(/usr/local/include/eigen3/)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "-g")

add_library(tdt
  tdt/gltf.cc
  tdt/gltf_entity.cc
  tdt/render_context.cc
  tdt/tdt.cc)

add_library(viewUtils
  tdt/window.cc)

add_executable(test_tdt
  test/test.cc)

target_link_libraries(viewUtils ${GL_LIBS})
target_link_libraries(test_tdt tdt viewUtils)
