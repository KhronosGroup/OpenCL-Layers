if (NOT TARGET OpenCL::HeadersCpp)
    find_package(OpenCLHeadersCpp REQUIRED)
endif()

add_library(ProgramCache
    src/program_cache.cpp)
target_include_directories(ProgramCache PUBLIC inc)
target_link_libraries(ProgramCache PUBLIC LayersCommon OpenCL::HeadersCpp)
target_compile_features(ProgramCache PUBLIC cxx_std_17)
target_compile_definitions(ProgramCache PUBLIC CL_HPP_TARGET_OPENCL_VERSION=300)

if (LAYERS_BUILD_TESTS)
    add_subdirectory(test)
endif()
