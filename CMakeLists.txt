cmake_minimum_required (VERSION 3.1)

project (OpenCL-Layers VERSION 0.1)
include (GNUInstallDirs)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_C_STANDARD 99)
set (CMAKE_C_STANDARD_REQUIRED ON)

set (OPENCL_ICD_LOADER_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/inc CACHE PATH "Path to OpenCL Headers")
include_directories (${OPENCL_ICD_LOADER_HEADERS_DIR})
add_definitions (-DCL_TARGET_OPENCL_VERSION=300)
if (NOT WIN32)
	add_compile_options(-Wall -Wextra -pedantic -Werror)
endif ()

add_subdirectory (simple-print)
add_subdirectory (ocl-icd-compat)
add_subdirectory (object-lifetime)
