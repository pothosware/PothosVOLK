########################################################################
# Project setup
########################################################################
cmake_minimum_required(VERSION 3.3)
project(PothosVOLK CXX)

find_package(Pothos 0.7 CONFIG REQUIRED)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake/Modules)

########################################################################
# Find VOLK
########################################################################
find_package(Volk REQUIRED)

########################################################################
# Find Python modules used to implement blocks
########################################################################
find_package(PythonInterp REQUIRED)
include(FindPythonModule)
find_python_module(mako REQUIRED)
find_python_module(yaml REQUIRED)

########################################################################
# Build module
########################################################################
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${CMAKE_CURRENT_SOURCE_DIR}/tests)

set(sources
    source/Module.cpp)

include(PothosUtil)
POTHOS_MODULE_UTIL(
    TARGET VOLKBlocks
    SOURCES
        ${sources}
    LIBRARIES
        Volk::volk
    DESTINATION volk
    ENABLE_DOCS ON
)
