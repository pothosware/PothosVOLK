########################################################################
# Project setup
########################################################################
cmake_minimum_required(VERSION 3.3)
project(PothosVOLK CXX)

find_package(Pothos 0.7 CONFIG REQUIRED)

########################################################################
# Find VOLK
########################################################################
find_package(Volk REQUIRED)

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
