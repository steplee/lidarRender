project(cumeshing LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.16)

#add_definitions(-g -O3 -fopenmp -std=c++17)
set(CMAKE_CXX_FLAGS "-g -O3 -fopenmp -std=c++17")
set(CMAKE_CUDA_FLAGS "-g -O3 --extended-lambda")

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

add_library(cumeshing
  #src/recon/dt.cc
  #src/recon/mfmc.cc
  #src/recon/torch_stuff.cc
  #src/recon/sorted_octree.cu
  #src/recon/visibility.cu
  #src/recon/octree_utils.cu
  #src/recon/vu.cu

  src/las/las_tiling.cc
  src/las/read_las.cc

  src/geo/geodset.cc

  src/viz/indexed_mesh.cc
  src/viz/window.cc
  src/viz/camera.cc
  src/viz/viz_app.cc

  src/heightmap/octree.cu
  src/heightmap/make.cu
  src/heightmap/make_geo.cu
  src/heightmap/marching_cubes.cu
  src/heightmap/surface_net.cu

  src/align/align.cu
  src/align/corr.cu
  )

target_link_libraries(cumeshing
  pthread GL GLEW glfw stdc++fs cufft
  ${LASZIP_LIBS}
  ${CGAL_LIBS}
  ${OpenCV_LIBS}
  ${GDAL_LIBRARIES}
  ${TORCH_LIBRARIES} /usr/local/cuda/lib64/libcusolver.so)

add_executable(testAlign src/align/test_align.cc)
target_link_libraries(testAlign cumeshing)

add_executable(mesh src/app/main.cc)
target_link_libraries(mesh cumeshing)

add_executable(test_mc src/app/test_mc.cc)
target_link_libraries(test_mc cumeshing)
