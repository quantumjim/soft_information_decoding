cmake_minimum_required(VERSION 3.5)
project(Soft_Info)

find_package(Python3 3.11 REQUIRED Interpreter Development)
add_subdirectory(libs/pybind11)

pybind11_add_module(example_module src/cpp/example.cpp)
pybind11_add_module(cpp_probabilities src/cpp/probabilities.cpp)
