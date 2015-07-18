cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

project("MachineLearningLibrary")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

enable_testing()

include(gmock)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(warnings "-Wall -Wextra -Werror")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(warnings "/W4 /WX /EHsc")
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

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  "c:/Local/include/eigen3"
  )

add_subdirectory(MLCore)

add_subdirectory(MLGaussian)

add_executable(
  LibMLMain
  src/main.cpp
  )

target_link_libraries(
  LibMLMain
  MLCore
  MLGaussian
  )

target_compile_features(
  LibMLMain PRIVATE cxx_range_for
  )

add_test(
  LibMLMainTest
  LibMLMain
  )

set(CONFIGURED_ONCE TRUE
  CACHE INTERNAL
  "A flag showing that CMake has configured at least once")