if (NOT TARGET OpenCL::HeadersCpp)
    find_package(OpenCLHeadersCpp REQUIRED)
endif()

find_package(Boost COMPONENTS wave)

add_library(ProgramCache
    src/program_cache.cpp
    src/preprocessor.cpp
    $<$<CXX_COMPILER_ID:MSVC>:src/instantiate_lexer.cpp>)

target_include_directories(ProgramCache PUBLIC inc)
target_link_libraries(ProgramCache PUBLIC LayersCommon OpenCL::HeadersCpp)
target_link_libraries(ProgramCache PRIVATE Boost::wave)
target_compile_features(ProgramCache PUBLIC cxx_std_17)
target_compile_definitions(ProgramCache PUBLIC
    CL_HPP_ENABLE_EXCEPTIONS
    CL_HPP_MINIMUM_OPENCL_VERSION=100
    CL_HPP_TARGET_OPENCL_VERSION=300)
target_compile_definitions(ProgramCache PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:BOOST_WAVE_SEPARATE_LEXER_INSTANTIATION=0>)
set_target_properties(ProgramCache PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (LAYERS_BUILD_TESTS)
    add_subdirectory(test)
endif()

install(TARGETS ProgramCache
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
