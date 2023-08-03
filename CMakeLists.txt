cmake_minimum_required(VERSION 3.5)
project(SECS_Divergence_Test)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_compile_options(-std=c++11 -Wall -O3 -fopenmp) 

add_subdirectory(${CMAKE_SOURCE_DIR}/medusa/ ${CMAKE_SOURCE_DIR}/binary/)
include_directories(${CMAKE_SOURCE_DIR}/medusa/include/)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/adaptive)
add_executable(divergence adaptive/divergence.cpp)
target_link_libraries(divergence medusa)
