# add_gmock_test(
#   mltest
#   test_mlcore.cpp
#   test_mlgaussian.cpp
#   )
# target_include_directories(mltest PRIVATE ${CMAKE_SOURCE_DIR}/ml)
# target_link_libraries(mltest ml)
# target_compile_features(mltest PRIVATE cxx_range_for)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# testlibml
project(testlibml VERSION 0.0.1 LANGUAGES CXX)

add_executable(
  testlibml
  src/test_mlcore.cpp
  src/test_mlgaussian.cpp
  )

target_include_directories(
  testlibml
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )

target_link_libraries(
  testlibml
  ml
  Eigen3::Eigen
  GTest::GTest
  GTest::Main
  )

add_test(
  libmltest
  testlibml
  )
