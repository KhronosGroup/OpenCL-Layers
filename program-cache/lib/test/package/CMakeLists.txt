cmake_minimum_required (VERSION 3.3)
cmake_policy(VERSION 3.3...3.22)

project(ProgramCachePackageTest CXX)
enable_testing()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/BoostWave.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/OpenCLHeaders.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/OpenCLHeadersCpp.cmake")

find_package(OpenCLICDLoader CONFIG REQUIRED)
find_package(ProgramCache CONFIG REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        OpenCL::ProgramCache
        OpenCL::HeadersCpp
        OpenCL::OpenCL
        Boost::wave
)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    CL_TARGET_OPENCL_VERSION=300
    CL_HPP_TARGET_OPENCL_VERSION=300
    $<$<AND:$<PLATFORM_ID:Windows>,$<OR:$<CXX_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:Clang>>>:NOMINMAX>
)
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
