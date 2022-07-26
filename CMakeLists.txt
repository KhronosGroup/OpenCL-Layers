cmake_minimum_required (VERSION 3.3)
cmake_policy(VERSION 3.3...3.22)

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

if(CMAKE_VERSION VERSION_LESS 3.7)
    add_library(OpenCL_lib INTERFACE)
    target_include_directories(OpenCL_lib INTERFACE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(OpenCL_lib INTERFACE ${OpenCL_LIBRARIES})
    add_library(OpenCL::OpenCL ALIAS OpenCL_lib)
endif()

set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Common layer flags
add_library (LayersCommon INTERFACE)

# C and C++ standard,
if (CMAKE_VERSION VERSION_LESS 3.8)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
else()
    target_compile_features(LayersCommon INTERFACE c_std_11 cxx_std_14)
endif()

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

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR OPENCL_LAYERS_BUILD_TESTING)
    include(CTest)
endif()
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR OPENCL_LAYERS_BUILD_TESTING) AND BUILD_TESTING)
    set(LAYERS_BUILD_TESTS ON)
endif()

if (LAYERS_BUILD_TESTS)
    if (NOT TARGET OpenCL)
        if(APPLE) # Apple's own loader can't load layers. We have to avoid it.
          find_package (OpenCLICDLoader REQUIRED)
        else()
          find_package (OpenCL REQUIRED)
        endif()
    endif ()
endif ()
if ((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR OPENCL_LAYERS_BUILD_TESTING) AND BUILD_TESTING)
  set (LAYERS_BUILD_TESTS ON)
endif ()
if ((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR OPENCL_LAYERS_BUILD_TESTING) AND BUILD_TESTING)
  set (LAYERS_BUILD_TESTS ON)
endif ()

if(LAYERS_BUILD_TESTS)
    if (NOT TARGET OpenCL)
        if(APPLE) # Apple's own loader can't load layers. We have to avoid it.
          find_package (OpenCLICDLoader REQUIRED)
        else()
          find_package (OpenCL REQUIRED)
        endif()
    endif ()
endif ()

include (GNUInstallDirs)

list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(Dependencies)

add_subdirectory (simple-print)
add_subdirectory (ocl-icd-compat)
add_subdirectory (object-lifetime)
add_subdirectory (param-verification)
add_subdirectory (utils)
