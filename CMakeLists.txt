cmake_minimum_required(VERSION 3.8)
project(Soft_Info)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Python3 3.11 EXACT REQUIRED Interpreter Development)
add_subdirectory(libs/pybind11)
include_directories(libs/cpp/eigen-3.4.0)

find_package(Armadillo REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(MLPACK REQUIRED mlpack)

include_directories(${MLPACK_INCLUDE_DIRS})
link_directories(${MLPACK_LIBRARY_DIRS})
add_definitions(${MLPACK_CFLAGS_OTHER})

include_directories(${CMAKE_SOURCE_DIR}/libs/Stim/src)
include_directories(${CMAKE_SOURCE_DIR}/libs/PyMatching/src)
# include_directories(${CMAKE_SOURCE_DIR}/libs/PyMatching/Stim/src) # Add this line

include_directories(${CMAKE_SOURCE_DIR}/src/cpp/Probabilities)
include_directories(${CMAKE_SOURCE_DIR}/src/cpp/PyMatching)

pybind11_add_module(example_module ${CMAKE_SOURCE_DIR}/src/cpp/example.cpp)

# Pybind11 module for the cpp_soft_info
pybind11_add_module(cpp_soft_info
    ${CMAKE_SOURCE_DIR}/src/cpp/bindings.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/Probabilities/probabilities.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/Probabilities/KDE.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/PyMatching/matching_graph.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/PyMatching/user_graph_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/PyMatching/predecoders.cpp)  

# Link the PyMatching library
set(PYMATCHING_LIB "${CMAKE_SOURCE_DIR}/libs/PyMatching/bazel-bin")

# Link the library to your modules as needed
target_link_libraries(cpp_soft_info
    PRIVATE
    ${PYMATCHING_LIB}/liblibpymatching.a
    ${PYMATCHING_LIB}/external/stim/libstim_lib.a
    ${MLPACK_LIBRARIES}
    # Any other libraries you need to link
)