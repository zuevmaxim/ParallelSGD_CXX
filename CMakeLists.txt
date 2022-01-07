cmake_minimum_required(VERSION 3.16)
project(ParallelSGDcpp)

set(CMAKE_CXX_STANDARD 17)

add_executable(ParallelSGDcpp main.cpp)
set(CMAKE_CXX_FLAGS "-lpthread -Wall -O3 -march=native")
TARGET_LINK_LIBRARIES(ParallelSGDcpp
        pthread
        )