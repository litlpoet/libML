add_library(
  MLCore
  mathmatrixpredefined.cpp
  )
add_subdirectory(unit_test)
target_compile_features(
  MLCore PRIVATE cxx_range_for
  )
