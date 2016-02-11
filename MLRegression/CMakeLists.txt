add_library(
  MLRegression
  regression.cpp
  gaussianprocess.cpp
  )

target_link_libraries(
  MLRegression
  MLCore
  )

target_compile_features(
  MLRegression PRIVATE cxx_range_for
  )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
