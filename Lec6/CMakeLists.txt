cmake_minimum_required(VERSION 3.4.1)

project(sift_match)

set(CMAKE_BUILD_TYPE "Debug")
add_definitions(-std=c++11 -g -o0)

FILE(GLOB ALL_SOURCES 
    "./src/app/*.cpp" 
    "./src/base/*.cpp" 
    "./src/calib/*.cpp" 
    "./src/util/*.cpp" 
    "./src/feature/*.cpp"
)

FILE(GLOB ALL_INCLUDES 
    "./src/app/*.h" 
    "./src/app/*.hpp" 
    "./src/base/*.h" 
    "./src/base/*.hpp" 
    "./src/calib/*.h" 
    "./src/calib/*.hpp" 
    "./src/util/*.h" 
    "./src/util/*.hpp"
    "./src/feature/*.h" 
    "./src/feature/*.hpp"
)

set(ALL_SRCS ${ALL_SOURCES} ${ALL_INCLUDES})

find_package(OpenCV)
find_package(Eigen3)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Eigen3_INCLUDE_DIRS})
include_directories(src/)

add_library(mango ${ALL_SRCS})
target_link_libraries(mango ${OpenCV_LIBS})

add_executable(sift_match src/app/sift_match.cpp)
target_link_libraries(sift_match mango ${OpenCV_LIBS})