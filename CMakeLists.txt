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
find_package(Volk)
if(NOT VOLK_FOUND)
    message(STATUS "Could not find compatible VOLK version.")
    return()
endif()

########################################################################
# Build module
########################################################################
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${VOLK_INCLUDE_DIRS})

set(sources
    source/Accumulator.cpp
    source/AddQuad.cpp
    source/BlockFactories.cpp
    source/Byteswap.cpp
    source/ModRange.cpp
    source/Module.cpp
    source/Normalize.cpp
    source/PopCnt.cpp
    source/PowerSpectralDensity.cpp
    source/QuadMaxStar.cpp
    source/SharedBufferAllocator.cpp
    source/SquareDist.cpp

    tests/BlockTests.cpp)

include(PothosUtil)
POTHOS_MODULE_UTIL(
    TARGET VOLKBlocks
    SOURCES ${sources}
    LIBRARIES ${VOLK_LIBRARIES}
    DESTINATION volk
    ENABLE_DOCS ON
)

########################################################################
# Search for VOLK kernels added after minimum versions so we know to
# compile in fallback versions if needed.
########################################################################
include(CheckSymbolExists)
set(CMAKE_REQUIRED_INCLUDES ${VOLK_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${VOLK_LIBRARIES})

function(CheckSymbolAndSetDefine symbol header variable)
    check_symbol_exists(${symbol} ${header} ${variable})
    if(${${variable}})
        target_compile_definitions(VOLKBlocks PRIVATE -D${variable})
    endif()
endfunction()

CheckSymbolAndSetDefine(
    "volk_32fc_accumulator_s32fc"
    "volk/volk.h"
    HAVE_32FC_ACCUMULATOR)
CheckSymbolAndSetDefine(
    "volk_32f_s32f_add_32f"
    "volk/volk.h"
    HAVE_32F_S32F_ADD)
CheckSymbolAndSetDefine(
    "volk_32f_exp_32f"
    "volk/volk.h"
    HAVE_32F_EXP)
CheckSymbolAndSetDefine(
    "volk_32fc_x2_s32fc_multiply_conjugate_add_32fc"
    "volk/volk.h"
    HAVE_32FC_X2_S32FC_MULTIPLY_CONJUGATE_ADD)
