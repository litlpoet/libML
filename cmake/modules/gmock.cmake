set(GMOCK_DIR "../../Libraries/gmock-1.6.0"  # starting from the project root
  CACHE PATH
  "The path to the Google mock test framework")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  option(gtest_force_shared_crt
    "Use shared (DLL) run-time lib even when google test is built as static lib."
    ON)
elseif (APPLE)
  add_definitions(-DGTEST_USE_OWN_TR1_TUPLE=1)
endif()

add_subdirectory(${GMOCK_DIR} ${CMAKE_BINARY_DIR}/gmock)
set_property(TARGET gtest gmock gmock_main APPEND_STRING PROPERTY COMPILE_FLAGS " -w")

include_directories(
  SYSTEM
  ${GMOCK_DIR}/gtest/include
  ${GMOCK_DIR}/include)

# add_gmock_test(<target> <sources>...)
#  Adds a Google mock based test executable, <target>, build from <sources>
#  and adds the test so that CTest will run it. Both the executable and the
#  test will be named <target>
function(add_gmock_test target)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} gmock_main)
  add_test(${target} ${target})
  add_custom_command(TARGET ${target}
    POST_BUILD
    COMMAND ${target}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running ${target}" VERBATIM)
endfunction()
