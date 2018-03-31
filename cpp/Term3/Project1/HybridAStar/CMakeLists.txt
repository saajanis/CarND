cmake_minimum_required(VERSION 3.9)
project(HybridAStar)

set(CMAKE_CXX_STANDARD 11)

add_executable(HybridAStar main.cpp hybrid_breadth_first.cpp hybrid_breadth_first.h hybrid_a_star.cpp hybrid_a_star.h)