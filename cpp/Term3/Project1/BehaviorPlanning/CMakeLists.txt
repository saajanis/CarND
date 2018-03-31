cmake_minimum_required(VERSION 3.9)
project(BehaviorPlanning)

set(CMAKE_CXX_STANDARD 11)

add_executable(BehaviorPlanning main.cpp vehicle.cpp vehicle.h cost.cpp cost.h road.cpp road.h)