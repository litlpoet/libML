project(libml VERSION 0.0.1 LANGUAGES CXX)

add_library(
  ml
  src/core/mathmatrixpredefined.cpp
  src/core/timeseriesdata.cpp
  src/interpolation/interpolation.cpp
  src/interpolation/gaussianinterpolation.cpp
  src/interpolation/gaussianinterpolationnoisy.cpp
  src/interpolation/multilevelbsplineinterpolation.cpp
  src/regression/gaussianprocess.cpp
  src/regression/kernelfunction.cpp
  src/regression/kernelsquaredexponential.cpp
  src/regression/regression.cpp
  src/regression/trainingdataset.cpp
  src/regression/gpoptimizerrprop.cpp
  )

target_include_directories(
  ml
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )

target_compile_features(
  ml
  PRIVATE
  cxx_range_for
  )

target_link_libraries(
  ml
  Eigen3::Eigen
  )

# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
