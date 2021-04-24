project(meshing)
cmake_minimum_required(VERSION 3.16)

#add_definitions(-g -O3 -fopenmp -std=c++17)
add_definitions(-g -O1 -fopenmp -std=c++17)

set(LASZIP_LIBS "laszip")
set(CGAL_LIBS "gmp")
find_package(GDAL REQUIRED)

find_package(glfw3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  /opt/CGAL-5.1/include
  /usr/local/include/eigen3/
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  )

add_library(meshing
  src/recon/dt.cc
  src/las/las_tiling.cc
  src/las/read_las.cc
  src/geodset.cc
  )

add_executable(mesh src/main.cc)
target_link_libraries(meshing pthread GL GLEW glfw stdc++fs ${LASZIP_LIBS} ${CGAL_LIBS} ${OpenCV_LIBS} ${GDAL_LIBRARIES})
target_link_libraries(mesh meshing)
