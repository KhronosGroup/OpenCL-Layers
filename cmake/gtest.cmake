if (NOT DEPENDENCIES_FORCE_DOWNLOAD)
  find_package(GTest)
endif ()

if (NOT GTest_FOUND)
  cmake_minimum_required(VERSION 3.11)
  include (FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.12.0
  )
  FetchContent_MakeAvailable(googletest)
endif ()
