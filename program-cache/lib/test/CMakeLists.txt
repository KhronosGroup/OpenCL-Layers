add_executable(ProgramCacheTest
    test_program_cache.cpp
    test_preprocessor.cpp
    test_utils.cpp)
target_link_libraries(ProgramCacheTest PRIVATE
    ProgramCache
    OpenCL::OpenCL
    OpenCL::HeadersCpp
    gtest_with_main)
target_compile_features(ProgramCacheTest PRIVATE cxx_std_17)
target_compile_definitions(ProgramCacheTest PRIVATE
    CL_TARGET_OPENCL_VERSION=300
    CL_HPP_ENABLE_EXCEPTIONS
    CL_HPP_MINIMUM_OPENCL_VERSION=100
    CL_HPP_TARGET_OPENCL_VERSION=300)
add_test(NAME ProgramCacheTest COMMAND ProgramCacheTest)

set_target_properties (ProgramCacheTest
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
        PDB_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        FOLDER "Layers"
)
