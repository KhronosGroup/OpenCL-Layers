cmake_minimum_required(VERSION 3.16)
# Project name is deliberately short: avoid long paths problem on Windows
project(CodeGen CXX)
include("${CMAKE_MODULE_PATH}/RapidXml.cmake")

add_executable(${PROJECT_NAME} parser.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE RapidXml::RapidXml)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)
