project(meshing LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.16)

#add_definitions(-g -O3 -fopenmp -std=c++17)
set(CMAKE_CXX_FLAGS "-g -O3 -fopenmp -std=c++17")
set(CMAKE_CUDA_FLAGS "-g -O3")

set(LASZIP_LIBS "laszip")
set(CGAL_LIBS "gmp")
find_package(GDAL REQUIRED)

find_package(glfw3 REQUIRED)
find_package(OpenCV REQUIRED)

if (EXISTS "/opt/libtorch/share/cmake")
  set(CMAKE_PREFIX_PATH "/opt/libtorch/share/cmake")
else()
  execute_process(COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
endif()
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  /opt/CGAL-5.1/include
  /usr/local/include/eigen3/
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  )

add_library(meshing
  src/recon/dt.cc
  src/recon/mfmc.cc
  src/recon/torch_stuff.cc
  src/recon/sorted_octree.cu
  src/recon/visibility.cu
  src/recon/octree_utils.cu
  src/recon/vu.cu

  src/las/las_tiling.cc
  src/las/read_las.cc

  src/geodset.cc

  src/viz/indexed_mesh.cc
  src/viz/window.cc
  src/viz/camera.cc
  )

add_executable(mesh src/main.cc)
target_link_libraries(meshing
  pthread GL GLEW glfw stdc++fs
  ${LASZIP_LIBS}
  ${CGAL_LIBS}
  ${OpenCV_LIBS}
  ${GDAL_LIBRARIES}
  ${TORCH_LIBRARIES} cusolver)
target_link_libraries(mesh meshing)
