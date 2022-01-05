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

endif ()

add_subdirectory (simple-print)
add_subdirectory (ocl-icd-compat)
add_subdirectory (object-lifetime)
