if (NOT DEPENDENCIES_FORCE_DOWNLOAD)
  find_package(RapidXml)
endif ()

if (NOT RapidXml_FOUND)
  if (NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/_deps/rapidxml-external-src")
    if (DEPENDENCIES_FORCE_DOWNLOAD)
      message (STATUS "DEPENDENCIES_FORCE_DOWNLOAD is ON. Fetching RapidXml.")
    else ()
      message (STATUS "Fetching RapidXml.")
    endif ()
  endif ()
  cmake_minimum_required(VERSION 3.11)
  include (FetchContent)
  FetchContent_Declare (
    rapidxml-external
    URL https://kumisystems.dl.sourceforge.net/project/rapidxml/rapidxml/rapidxml%201.13/rapidxml-1.13.zip
    URL_HASH SHA256=c3f0b886374981bb20fabcf323d755db4be6dba42064599481da64a85f5b3571
  )
  FetchContent_MakeAvailable (rapidxml-external)
  list (APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_BINARY_DIR}/_deps/rapidxml-external-src")
  find_package (RapidXml REQUIRED)
endif ()
