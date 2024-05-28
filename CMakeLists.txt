cmake_minimum_required(VERSION 3.8)
project(Soft_Info)

# Set the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Use vcpkg toolchain file
set(CMAKE_TOOLCHAIN_FILE "C:/projects/soft_info/libs/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

# Find Python3
find_package(Python3 3.11 EXACT REQUIRED Interpreter Development)

# Add subdirectories
add_subdirectory(libs/pybind11)

# Include directories for dependencies
include_directories(libs/cpp/eigen-3.4.0)

# Find Armadillo
find_package(Armadillo REQUIRED)
if(NOT Armadillo_FOUND)
    set(ARMADILLO_INCLUDE_DIRS "C:/projects/soft_info/libs/vcpkg/installed/x64-windows/include")
    set(ARMADILLO_LIBRARIES "C:/projects/soft_info/libs/vcpkg/installed/x64-windows/lib/armadillo.lib")
    include_directories(${ARMADILLO_INCLUDE_DIRS})
    link_directories(${ARMADILLO_LIBRARIES})
endif()
message(STATUS "Armadillo Include Dir: ${ARMADILLO_INCLUDE_DIRS}")
message(STATUS "Armadillo Libraries: ${ARMADILLO_LIBRARIES}")

# Include mlpack as a header-only library
include_directories("C:/projects/soft_info/libs/vcpkg/installed/x64-windows/include")
message(STATUS "MLPACK Include Dir: C:/projects/soft_info/libs/vcpkg/installed/x64-windows/include")

# Include directories for other dependencies
include_directories(${CMAKE_SOURCE_DIR}/libs/Stim/src)
include_directories(${CMAKE_SOURCE_DIR}/libs/PyMatching/src)

include_directories(${CMAKE_SOURCE_DIR}/src/cpp/Probabilities)
include_directories(${CMAKE_SOURCE_DIR}/src/cpp/PyMatching)
include_directories(${CMAKE_SOURCE_DIR}/src/cpp/StimUtils)

# Enable OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
endif()

# Pybind11 module for the cpp_soft_info
pybind11_add_module(cpp_soft_info
    ${CMAKE_SOURCE_DIR}/src/cpp/bindings.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/Probabilities/probabilities.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/Probabilities/KDE.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/Probabilities/convertor.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/PyMatching/matching_graph.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/PyMatching/user_graph_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/PyMatching/predecoders.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/PyMatching/convdecoders.cpp  
    ${CMAKE_SOURCE_DIR}/src/cpp/StimUtils/utils.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/StimUtils/StimDecoder.cpp)  

# Link the PyMatching library
set(PYMATCHING_LIB "${CMAKE_SOURCE_DIR}/libs/PyMatching/bazel-bin")

# Link the library to your modules as needed
target_link_libraries(cpp_soft_info
    PRIVATE
    # ${PYMATCHING_LIB}/liblibpymatching.a
    # ${PYMATCHING_LIB}/external/stim/libstim_lib.a   
    ${PYMATCHING_LIB}/libpymatching.lib
    ${PYMATCHING_LIB}/external/stim/stim_lib.lib   
    ${ARMADILLO_LIBRARIES}
    # Any other libraries you need to link
)



# Print diagnostic information
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
