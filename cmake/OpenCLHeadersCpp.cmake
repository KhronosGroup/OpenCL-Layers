if (NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(OpenCLHeadersCpp)
endif ()

if (NOT OpenCLHeadersCpp_FOUND)
  if (DEPENDENCIES_FORCE_DOWNLOAD)
    message (STATUS "DEPENDENCIES_FORCE_DOWNLOAD is ON. Fetching OpenCLHeadersCpp")
  else ()
    message (STATUS "Fetching OpenCLHeadersCpp")
  endif ()
  include (FetchContent)
  FetchContent_Declare(
    OpenCLHeadersCpp
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git
    GIT_TAG        v2023.04.17
  )

  # A workaround to disable the building of the examples
  # which requires the OpenCL-ICD-Loader CMake project to be installed.
  if(DEFINED CACHE{BUILD_EXAMPLES})
    set(BUILD_EXAMPLES_CACHE_VAL "${BUILD_EXAMPLES}")
  endif()
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(OpenCLHeadersCpp)

  # Restoring the original state
  if(DEFINED BUILD_EXAMPLES_CACHE_VAL)
    set(BUILD_EXAMPLES "${BUILD_EXAMPLES_CACHE_VAL}" CACHE BOOL "" FORCE)
  else()
    unset(BUILD_EXAMPLES CACHE)
  endif()
endif ()
