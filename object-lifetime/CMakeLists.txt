set (OPENCL_OBJECT_LIFETIME_LAYER_SOURCES
    object_lifetime.cpp)

if (WIN32)
    list (APPEND OPENCL_OBJECT_LIFETIME_LAYER_SOURCES object_lifetime.def)
else ()
    if (NOT APPLE)
        list (APPEND OPENCL_OBJECT_LIFETIME_LAYER_SOURCES object_lifetime.map)
    endif ()
endif ()

add_library (CLObjectLifetimeLayer SHARED ${OPENCL_OBJECT_LIFETIME_LAYER_SOURCES})

if (NOT WIN32 AND NOT APPLE)
  set_target_properties (CLObjectLifetimeLayer PROPERTIES LINK_FLAGS "-Wl,--version-script -Wl,${CMAKE_CURRENT_SOURCE_DIR}/object_lifetime.map")
endif ()

install (TARGETS CLObjectLifetimeLayer
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
