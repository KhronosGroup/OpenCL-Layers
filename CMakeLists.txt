cmake_minimum_required (VERSION 3.8) # cxx_std_17

# Include guard for including this project multiple times
if (TARGET PrintLayer)
    return ()
endif ()

project (OpenCL-Layers
    VERSION 0.1
    LANGUAGES C CXX
)

set (OPENCL_ICD_LOADER_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/inc CACHE PATH "Path to OpenCL Headers")
option (OPENCL_LAYERS_BUILD_TESTING "Enable support for OpenCL layers testing." OFF)

# Search for dependencies
if (EXISTS ${OPENCL_ICD_LOADER_HEADERS_DIR}/CL/cl.h)
    if (NOT TARGET OpenCL::Headers)
        message (STATUS "Defining OpenCL::Headers through OPENCL_ICD_LOADER_HEADERS_DIR")
    endif  ()
    add_library (OpenCLHeaders INTERFACE)
    add_library (OpenCL::Headers ALIAS OpenCLHeaders)
    target_include_directories (OpenCLHeaders INTERFACE ${OPENCL_ICD_LOADER_HEADERS_DIR})
else ()
    if (NOT TARGET OpenCL::Headers)
        find_package (OpenCLHeaders REQUIRED)
    endif ()
endif ()

if(OPENCL_LAYERS_BUILD_TESTING)
    if (NOT TARGET OpenCL)
        find_package (OpenCL REQUIRED)
    endif ()
endif ()

# Common layer flags
add_library (LayersCommon INTERFACE)

target_link_libraries (LayersCommon INTERFACE OpenCL::Headers)

target_compile_features (LayersCommon
    INTERFACE
        c_std_11
        cxx_std_17
)

target_compile_definitions (LayersCommon
    INTERFACE
        CL_TARGET_OPENCL_VERSION=300
        CL_USE_DEPRECATED_OPENCL_1_0_APIS
        CL_USE_DEPRECATED_OPENCL_1_1_APIS
        CL_USE_DEPRECATED_OPENCL_1_2_APIS
        CL_USE_DEPRECATED_OPENCL_2_0_APIS
        CL_USE_DEPRECATED_OPENCL_2_1_APIS
        CL_USE_DEPRECATED_OPENCL_2_2_APIS
)

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR OPENCL_LAYERS_BUILD_TESTING)
    include (CTest)
endif ()

include (GNUInstallDirs)

add_subdirectory (simple-print)
add_subdirectory (ocl-icd-compat)
add_subdirectory (object-lifetime)

set (INSTALL_TARGETS
    PrintLayer
    OclIcdCompatLayer
    CLObjectLifetimeLayer
)
set (BUILD_TARGETS ${INSTALL_TARGETS})
if (OPENCL_LAYERS_BUILD_TESTING)
    list (APPEND BUILD_TARGETS
        PrintLayerTest
        #OclIcdCompatLayerTest
        #CLObjectLifetimeLayerTest
    )
endif ()

set_target_properties (${BUILD_TARGETS}
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
        PDB_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        FOLDER "Layers/${OPENCL_SAMPLE_CATEGORY}/${OPENCL_SAMPLE_TARGET}"
)
