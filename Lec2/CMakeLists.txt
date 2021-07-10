cmake_minimum_required(VERSION 3.4.1)

project(draw_cube)

add_definitions(-std=c++11)

aux_source_directory(./src DIR_SRCS)

find_package(OpenCV)
find_package(Eigen3)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Eigen3_INCLUDE_DIRS})

add_executable(draw_cube ${DIR_SRCS})
target_link_libraries( draw_cube ${OpenCV_LIBS} )