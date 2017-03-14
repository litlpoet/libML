add_gmock_test(
  mltest
  test_mlcore.cpp
  test_mlgaussian.cpp
  )
target_include_directories(mltest PRIVATE ${CMAKE_SOURCE_DIR}/ml)
target_link_libraries(mltest ml)
target_compile_features(mltest PRIVATE cxx_range_for)
