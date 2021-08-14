cmake_minimum_required(VERSION 3.4.1)

project(stereo_vision)

add_definitions(-std=c++14 -g)

FILE(GLOB ALL_SOURCES 
    "./src/base/*.cpp" 
    "./src/calib/*.cpp" 
    "./src/common/*.cpp"
    "./src/util/*.cpp" 
    "./src/feature/*.cpp"
    "./src/vo/*.cpp"
)

FILE(GLOB ALL_INCLUDES 
    "./src/base/*.h" 
    "./src/base/*.hpp" 
    "./src/calib/*.h" 
    "./src/calib/*.hpp" 
    "./src/util/*.h" 
    "./src/util/*.hpp"
    "./src/feature/*.h" 
    "./src/feature/*.hpp"
    "./src/common/*.h"
    "./src/common/*.hpp"
    "./src/vo/*.h"
    "./src/vo/*.hpp"
)

set(ALL_SRCS ${ALL_SOURCES} ${ALL_INCLUDES})

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMAX_OPENMP_NUM_THREADS=${PROCESSOR_COUNT} -DOPENMP_NUM_THREADS=${PROCESSOR_COUNT} ${OpenMP_CXX_FLAGS} -DOPENMP")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Eigen3_INCLUDE_DIRS})
include_directories(src/)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(poly_ransac src/app/poly_ransac.cpp ${ALL_SRCS})
target_link_libraries(poly_ransac ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(stereo_p3p src/app/stereo_p3p.cpp ${ALL_SRCS})
target_link_libraries(stereo_p3p ${OpenCV_LIBS} ${PCL_LIBRARIES})