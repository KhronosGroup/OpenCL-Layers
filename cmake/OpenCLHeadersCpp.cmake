
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
  FetchContent_MakeAvailable(OpenCLHeadersCpp)
endif ()
