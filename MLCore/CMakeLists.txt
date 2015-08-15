add_library(
  MLCore
  mathmatrixpredefined.cpp
  timeseriesdata.cpp
  )
add_subdirectory(unit_test)
target_compile_features(
  MLCore PRIVATE cxx_range_for
  )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
