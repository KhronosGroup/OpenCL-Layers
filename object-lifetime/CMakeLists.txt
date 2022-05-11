add_library (CLObjectLifetimeLayer SHARED
    object_lifetime.cpp
    $<$<CXX_COMPILER_ID:MSVC>:object_lifetime.def>
    $<$<CXX_COMPILER_ID:GNU>:object_lifetime.map>
)

target_link_libraries (CLObjectLifetimeLayer PRIVATE LayersUtils LayersCommon)

if (NOT WIN32 AND NOT APPLE)
    set_target_properties (CLObjectLifetimeLayer PROPERTIES LINK_FLAGS "-Wl,--version-script -Wl,${CMAKE_CURRENT_SOURCE_DIR}/object_lifetime.map")
endif ()

set (INSTALL_TARGETS CLObjectLifetimeLayer)
set (BUILD_TARGETS ${INSTALL_TARGETS})

if (LAYERS_BUILD_TESTS)
    add_library (CLObjectLifetimeICD SHARED
        object_lifetime_test_icd.hpp
        object_lifetime_test_icd.cpp
        object_lifetime_test_icd_surface.cpp
        $<$<CXX_COMPILER_ID:MSVC>:object_lifetime_test_icd.def>
        $<$<CXX_COMPILER_ID:GNU>:object_lifetime_test_icd.map>
    )
    target_link_libraries (CLObjectLifetimeICD
        PRIVATE
            LayersCommon
            LayersUtils
            OpenCL::Headers
    )

    add_subdirectory (test)
endif ()

list (APPEND BUILD_TARGETS
    CLObjectLifetimeICD
)

set_target_properties (${BUILD_TARGETS}
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
        PDB_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        FOLDER "Layers"
)
install (
    TARGETS ${INSTALL_TARGETS}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
