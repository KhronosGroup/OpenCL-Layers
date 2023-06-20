add_executable(ProgramCacheTest test_program_cache.cpp)
target_link_libraries(ProgramCacheTest PUBLIC ProgramCache OpenCL::OpenCL gtest gtest_main)
add_test(NAME ProgramCacheTest COMMAND ProgramCacheTest)
