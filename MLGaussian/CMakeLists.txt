add_library(
  MLGaussian
  gaussianinterpolation.cpp
  )
add_subdirectory(unit_test)
target_compile_features(
  MLGaussian PRIVATE cxx_range_for
  )
