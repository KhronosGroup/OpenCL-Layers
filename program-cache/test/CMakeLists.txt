if (NOT TARGET OpenCL::HeadersCpp)
    find_package(OpenCLHeadersCpp REQUIRED)
endif()

add_executable(ProgramCacheLayerTest main.cpp)
target_link_libraries(ProgramCacheLayerTest PRIVATE LayersCommon OpenCL::OpenCL OpenCL::HeadersCpp)
target_compile_definitions(ProgramCacheLayerTest PRIVATE
    CL_HPP_ENABLE_EXCEPTIONS
    CL_HPP_MINIMUM_OPENCL_VERSION=100
    CL_HPP_TARGET_OPENCL_VERSION=300)

add_test(NAME ProgramCacheLayerTest COMMAND ProgramCacheLayerTest)
set_tests_properties(ProgramCacheLayerTest PROPERTIES ENVIRONMENT "OPENCL_LAYERS=$<TARGET_FILE:ProgramCacheLayer>")
