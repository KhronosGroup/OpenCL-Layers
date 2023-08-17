if (NOT DEPENDENCIES_FORCE_DOWNLOAD)
  find_package(GTest CONFIG)
  if (NOT GTest_FOUND)
    find_package(GTest)
  endif()
endif ()

if (NOT GTest_FOUND)
  if (DEPENDENCIES_FORCE_DOWNLOAD)
    message (STATUS "DEPENDENCIES_FORCE_DOWNLOAD is ON. Fetching googletest")
  else ()
    message (STATUS "Fetching googletest")
  endif ()
  include (FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.12.0
  )
  FetchContent_MakeAvailable(googletest)
endif ()
add_library(gtest_with_main INTERFACE)
if (TARGET GTest::GTest AND TARGET GTest::Main)
  target_link_libraries(gtest_with_main INTERFACE GTest::GTest GTest::Main)
elseif(TARGET GTest::gtest AND TARGET GTest::gtest_main)
  target_link_libraries(gtest_with_main INTERFACE GTest::gtest GTest::gtest_main)
else()
  message(FATAL_ERROR "Did not find gtest targets")
endif()
