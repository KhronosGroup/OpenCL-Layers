# - Find RapidXml
# Find the RapidXml headers
#
# RapidXml             - INTERFACE library to link against
# RapidXml_INCLUDE_DIR - where to find the RapidXml headers
# RapidXml_FOUND       - True if RapidXml was found

if (RapidXML_INCLUDE_DIR)
  # Already in cache
  set (RapidXML_FOUND TRUE)
endif ()

# Find the headers
find_path (
  RapidXml_INCLUDE_PATH
  PATHS
    ${RapidXml_DIR}
  NAMES
    rapidxml.hpp
  PATH_SUFFIXES
    inc
    include
    rapidxml
)

# handle the QUIETLY and REQUIRED arguments and set RapidXml_FOUND to TRUE
# if all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (
  RapidXml
  "RapidXml (https://sourceforge.net/projects/rapidxml/) not found"
  RapidXml_INCLUDE_PATH
)

if (RapidXml_FOUND)
  set (RapidXml_INCLUDE_DIR ${RapidXml_INCLUDE_PATH})
  add_library (RapidXml::RapidXml INTERFACE IMPORTED)
  target_include_directories (RapidXml::RapidXml INTERFACE ${RapidXml_INCLUDE_PATH})
endif ()

mark_as_advanced(RapidXml_INCLUDE_PATH)
