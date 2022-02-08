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
        if(APPLE) # Apple's own loader can't load layers. We have to avoid it.
          find_package (OpenCLICDLoader REQUIRED)
        else()
          find_package (OpenCL REQUIRED)
        endif()
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
