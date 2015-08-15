add_library(
  MLInterpolation
  interpolation.cpp
  gaussianinterpolation.cpp
  gaussianinterpolationnoisy.cpp
  multilevelbsplineinterpolation.cpp
  )

add_subdirectory(unit_test)

target_link_libraries(
  MLInterpolation
  MLCore
  )

target_compile_features(
  MLInterpolation PRIVATE cxx_range_for
  )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
