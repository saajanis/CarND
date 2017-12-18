cmake_minimum_required(VERSION 3.8)
project(CarND_Project7UKF)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES src/main.cpp src/ukf.h src/ukf.cpp)
add_executable(CarND_Project7UKF ${SOURCE_FILES})