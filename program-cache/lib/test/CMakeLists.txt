add_executable(ProgramCacheTest
    test_program_cache.cpp
    test_preprocessor.cpp
    test_utils.cpp)
target_link_libraries(ProgramCacheTest PRIVATE
    ProgramCache
    OpenCL::OpenCL
    OpenCL::HeadersCpp
    gtest
    gtest_main)
target_compile_features(ProgramCacheTest PRIVATE cxx_std_17)
target_compile_definitions(ProgramCacheTest PRIVATE
    CL_HPP_ENABLE_EXCEPTIONS
    CL_HPP_MINIMUM_OPENCL_VERSION=100
    CL_HPP_TARGET_OPENCL_VERSION=300)
add_test(NAME ProgramCacheTest COMMAND ProgramCacheTest)
