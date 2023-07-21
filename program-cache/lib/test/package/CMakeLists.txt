cmake_minimum_required (VERSION 3.3)
cmake_policy(VERSION 3.3...3.22)

project(ProgramCachePackageTest CXX)
enable_testing()
find_package(OpenCL CONFIG REQUIRED)
find_package(OpenCLHeadersCpp CONFIG REQUIRED)
find_package(ProgramCache CONFIG REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::ProgramCache OpenCL::HeadersCpp OpenCL::OpenCL)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    CL_TARGET_OPENCL_VERSION=300
    CL_HPP_TARGET_OPENCL_VERSION=300)
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
