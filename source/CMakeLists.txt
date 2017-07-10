# Top level CMakeLists
cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)
project(MachineLearningLibrary VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)

# list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
## ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules :
##     everytime ~/VersionControl/Modules/libML/cmake/modules


# include(gmock)

# if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
#     "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#   set(warnings "-Wall -Wextra")
# elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
#   set(warnings "/W4 /EHsc")
# endif()

# if (NOT CONFIGURED_ONCE)
#   set(CMAKE_CXX_FLAGS "${warnings}"
#     CACHE STRING
#     "C++ flags used by the compiler during all build types."
#     FORCE)
#   set(CMAKE_C_FLAGS "${warnings}"
#     CACHE STRING
#     "C flags used by the compiler during all build types."
#     FORCE)
# endif()

# if (WIN32)
#   set(EIGEN3_DIR "c:/Local/include/eigen3")
# else()
#   set(EIGEN3_DIR "/usr/include/eigen3")
# endif()

# message("eigen3 dir: " ${EIGEN3_DIR})
# include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_DIR})

enable_testing()
add_subdirectory(libml)
add_subdirectory(libml_test)

# set(CONFIGURED_ONCE TRUE CACHE INTERNAL
#   "A flag showing that CMake has configured at least once")
