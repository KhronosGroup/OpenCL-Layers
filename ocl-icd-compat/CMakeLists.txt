set (OPENCL_OCL_ICD_COMPAT_LAYER_SOURCES
    ocl_icd_compat.c)

if (WIN32)
    list (APPEND OPENCL_OCL_ICD_COMPAT_LAYER_SOURCES ocl_icd_compat.def)
else ()
    if (NOT APPLE)
        list (APPEND OPENCL_OCL_ICD_COMPAT_LAYER_SOURCES ocl_icd_compat.map)
    endif ()
endif ()

add_library (OclIcdCompatLayer SHARED ${OPENCL_OCL_ICD_COMPAT_LAYER_SOURCES})

if (NOT WIN32 AND NOT APPLE)
  set_target_properties (OclIcdCompatLayer PROPERTIES LINK_FLAGS "-Wl,--version-script -Wl,${CMAKE_CURRENT_SOURCE_DIR}/ocl_icd_compat.map")
endif ()

install (TARGETS OclIcdCompatLayer
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
