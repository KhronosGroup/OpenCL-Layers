cmake_minimum_required (VERSION 3.16)
cmake_policy(VERSION 3.16...3.26)

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

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR OPENCL_LAYERS_BUILD_TESTING)
    include(CTest)
endif()
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR OPENCL_LAYERS_BUILD_TESTING) AND BUILD_TESTING)
    set(LAYERS_BUILD_TESTS ON)
endif()

# Search for dependencies
list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(Dependencies)

set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Common layer flags
add_library (LayersCommon INTERFACE)

# C and C++ standard,
target_compile_features(LayersCommon INTERFACE c_std_11 cxx_std_14)

target_link_libraries (LayersCommon INTERFACE OpenCL::Headers)

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

if (LAYERS_BUILD_TESTS)
    if (NOT TARGET OpenCL)
        if(APPLE) # Apple's own loader can't load layers. We have to avoid it.
          find_package (OpenCLICDLoader REQUIRED)
        else()
          find_package (OpenCL REQUIRED)
        endif()
    endif ()
endif ()

include (GNUInstallDirs)

add_subdirectory (simple-print)
add_subdirectory (ocl-icd-compat)
add_subdirectory (object-lifetime)
add_subdirectory (param-verification)
add_subdirectory (program-cache)
add_subdirectory (utils)
