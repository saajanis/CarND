cmake_minimum_required(VERSION 3.9)
project(GNB)

set(CMAKE_CXX_STANDARD 11)

add_executable(GNB main.cpp classifier.h classifier.cpp)