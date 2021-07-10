cmake_minimum_required(VERSION 3.4.1)

project(calib)

add_definitions(-std=c++11 -g)

FILE(GLOB ALL_SOURCES "./src/app/*.cpp" "./src/base/*.cpp" "./src/calib/*.cpp" "./src/util/*.cpp")
FILE(GLOB ALL_INCLUDES "./src/app/*.h" "./src/app/*.hpp" "./src/base/*.h" "./src/base/*.hpp" "./src/calib/*.h" "./src/calib/*.hpp" "./src/util/*.h" "./src/util/*.hpp")

set(ALL_SRCS ${ALL_SOURCES} ${ALL_INCLUDES})

find_package(OpenCV)
find_package(Eigen3)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Eigen3_INCLUDE_DIRS})

add_executable(calib ${ALL_SRCS})
target_link_libraries(calib ${OpenCV_LIBS} )