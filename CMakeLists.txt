cmake_minimum_required(VERSION 3.17)
project(ParallelSGDcpp)

set(CMAKE_CXX_STANDARD 17)

add_executable(ParallelSGDcpp main.cpp)
set(CMAKE_CXX_FLAGS "-Wall -O3 -march=native")