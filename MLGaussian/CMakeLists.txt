add_library(
  MLGaussian
  gaussianinterpolation.cpp
  )

add_subdirectory(unit_test)

target_link_libraries(
  MLGaussian
  MLCore
  )

target_compile_features(
  MLGaussian PRIVATE cxx_range_for
  )
