add_executable(ProgramCacheLayerTest main.cpp)
target_link_libraries(ProgramCacheLayerTest PRIVATE LayersCommon OpenCL::OpenCL OpenCL::HeadersCpp)
target_compile_definitions(ProgramCacheLayerTest PRIVATE
    CL_HPP_ENABLE_EXCEPTIONS
    CL_HPP_MINIMUM_OPENCL_VERSION=100
    CL_HPP_TARGET_OPENCL_VERSION=300)

add_test(NAME ProgramCacheLayerTest COMMAND ProgramCacheLayerTest)
set_tests_properties(ProgramCacheLayerTest PROPERTIES ENVIRONMENT "OPENCL_LAYERS=$<TARGET_FILE:ProgramCacheLayer>")

set_target_properties (ProgramCacheLayerTest
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
        PDB_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        FOLDER "Layers"
)
