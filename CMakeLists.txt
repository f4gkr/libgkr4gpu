#****************************************************************
# *                                                              *
# * @copyright  Copyright (c) Sylvain Azarian - F4GKR            *
# * @author     Sylvain AZARIAN - s.azarian@sdr-technologies.fr  *
# * @project    GPU DDC Bank                                     *
# *                                                              *
# * Licence: GPL 3.0                                             *
# *                                                              *
# ****************************************************************

message( STATUS "CMake Script (c) Sylvain AZARIAN - F4GKR" )
cmake_minimum_required(VERSION 3.1)

project(ddc
    VERSION 1.0
    DESCRIPTION "DDC Cmake"
    LANGUAGES CXX)

option(OPTIMIZE_TARGET  "Build native -march=native" ON)

EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
message( STATUS "Architecture: ${ARCHITECTURE}" )


if( OPTIMIZE_TARGET )
    message(STATUS "Testing native support for compiler")

    if( ${ARCHITECTURE} STREQUAL "x86_64" )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mpclmul -msse2")
        add_definitions(-D__HAS_AVX__)
        message(STATUS "DDC: AVX2 CPU Features")
    else()
        include(CheckCXXCompilerFlag)
        CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
            message(STATUS "Native compile !")
        endif()
    endif()
endif()

include_directories(lib)
set(GPU_SOURCES "")
set(LIB_SOURCES "")
set(SOURCES "")

list(APPEND LIB_SOURCES
    lib/fircoeffcache.cpp
    lib/olasgpu.cpp
    lib/olasgpuchannel.cpp 
)

list(APPEND SOURCES
    example.cpp
)

SET(CMAKE_CUDA_COMPILER nvcc)
enable_language(CUDA)
list( APPEND GPU_SOURCES
        lib/cuda/zero.cu
)
list (APPEND SOURCES ${LIB_SOURCES}  ${GPU_SOURCES})
include_directories(/usr/local/cuda/include)
add_executable(ddcexample ${SOURCES})
target_compile_features(ddcexample PRIVATE cxx_std_14)

set_target_properties(
    ddcexample
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(ddcexample PROPERTIES CUDA_ARCHITECTURES "60;72")
target_link_libraries(ddcexample cudart)
target_link_libraries(ddcexample cufft)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

target_compile_options( ddcexample PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --default-stream per-thread
                       >)