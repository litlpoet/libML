cmake_minimum_required(VERSION 2.8.0 FATAL_ERROR)
project("MachineLearningLibrary")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
## ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules :
##     everytime ~/VersionControl/Modules/libML/cmake/modules

enable_testing()

include(gmock)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(warnings "-Wall -Wextra")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(warnings "/W4 /EHsc")
endif()

if (NOT CONFIGURED_ONCE)
  set(CMAKE_CXX_FLAGS "${warnings}"
    CACHE STRING
    "C++ flags used by the compiler during all build types."
    FORCE)
  set(CMAKE_C_FLAGS "${warnings}"
    CACHE STRING
    "C flags used by the compiler during all build types."
    FORCE)
endif()

if (WIN32)
  set(EIGEN3_DIR "c:/Local/include/eigen3")
else()
  set(EIGEN3_DIR "/usr/include/eigen3")
endif()

message("eigen3 dir: " ${EIGEN3_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_DIR})

add_subdirectory(ml)
add_subdirectory(ml_test)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CONFIGURED_ONCE TRUE CACHE INTERNAL
  "A flag showing that CMake has configured at least once")
