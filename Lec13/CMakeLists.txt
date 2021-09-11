cmake_minimum_required(VERSION 3.4.1)

project(ba)

add_definitions(-std=c++14 -O3)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

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
find_package(G2O REQUIRED)
Find_Package(Cholmod REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMAX_OPENMP_NUM_THREADS=${PROCESSOR_COUNT} -DOPENMP_NUM_THREADS=${PROCESSOR_COUNT} ${OpenMP_CXX_FLAGS} -DOPENMP")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Eigen3_INCLUDE_DIRS})
include_directories(${G2O_INCLUDE_DIRS})
include_directories(${CHOLMOD_INCLUDE_DIR})
include_directories(src/)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear 
    g2o_types_icp g2o_types_slam2d g2o_types_sba g2o_types_slam3d g2o_core g2o_interface 
    g2o_solver_csparse g2o_solver_structure_only g2o_csparse_extension g2o_opengl_helper g2o_solver_dense 
    g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )

add_executable(ba src/app/ba.cpp ${ALL_SRCS})
target_link_libraries(ba ${OpenCV_LIBS} ${PCL_LIBRARIES} ${G2O_LIBS} ${CHOLMOD_LIBRARIES})