cmake_minimum_required(VERSION 3.8)
project(Soft_Info)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Python3 3.11 REQUIRED Interpreter Development)
add_subdirectory(libs/pybind11)
include_directories(libs/cpp/eigen-3.4.0)
include_directories(libs/cpp/indicators-master/include)

pybind11_add_module(example_module src/cpp/example.cpp)
pybind11_add_module(cpp_probabilities src/cpp/probabilities.cpp)
