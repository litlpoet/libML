add_library(
  MLRegression
  gaussianprocess.cpp
  kernelfunction.cpp
  kernelsquaredexponential.cpp
  regression.cpp
  trainingdataset.cpp
  )

target_link_libraries(
  MLRegression
  MLCore
  )

target_compile_features(
  MLRegression PRIVATE cxx_range_for
  )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
