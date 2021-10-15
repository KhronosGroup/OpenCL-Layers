set (OPENCL_OBJECT_ACQUIRE_LAYER_SOURCES
    object_acquire.cpp)

if (WIN32)
    list (APPEND OPENCL_OBJECT_ACQUIRE_LAYER_SOURCES object_acquire.def)
else ()
    if (NOT APPLE)
        list (APPEND OPENCL_OBJECT_ACQUIRE_LAYER_SOURCES object_acquire.map)
    endif ()
endif ()

add_library (CLObjectAcquireLayer SHARED ${OPENCL_OBJECT_ACQUIRE_LAYER_SOURCES})

if (NOT WIN32 AND NOT APPLE)
  set_target_properties (CLObjectAcquireLayer PROPERTIES LINK_FLAGS "-Wl,--version-script -Wl,${CMAKE_CURRENT_SOURCE_DIR}/object_acquire.map")
endif ()

install (TARGETS CLObjectAcquireLayer
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
