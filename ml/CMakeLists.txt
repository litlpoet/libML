add_library(
  ml
  core/mathmatrixpredefined.cpp
  core/timeseriesdata.cpp
  interpolation/interpolation.cpp
  interpolation/gaussianinterpolation.cpp
  interpolation/gaussianinterpolationnoisy.cpp
  interpolation/multilevelbsplineinterpolation.cpp
  regression/gaussianprocess.cpp
  regression/kernelfunction.cpp
  regression/kernelsquaredexponential.cpp
  regression/regression.cpp
  regression/trainingdataset.cpp
  regression/gpoptimizerrprop.cpp
  )
target_include_directories(ml PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_feature(ml PRIVATE cxx_range_for)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
