# Top level CMakeLists
cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)
project(libML VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)

enable_testing()
add_subdirectory(libml)
add_subdirectory(libml_test)
