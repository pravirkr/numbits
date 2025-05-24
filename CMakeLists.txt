cmake_minimum_required(VERSION 3.18)

project(
  numbits
  VERSION 0.0.4
  LANGUAGES CXX
  DESCRIPTION "Pack and unpack 1, 2 and 4 bit data to/from 8-bit arrays.")

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

# Configuration options
option(BUILD_PYTHON "Build Python bindings" ON)
option(ENABLE_NATIVE "Enable native CPU optimizations (-march=native)" ON)
option(ENABLE_FAST_MATH "Enable fast math optimizations (-ffast-math)" ON)
option(ENABLE_IPO
       "Enable Interprocedural Optimization (Link-Time Optimization)" ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Release if unspecified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Defaulting to Release build.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

# IPO / LTO
if(CMAKE_BUILD_TYPE MATCHES "Release" AND ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported)
  if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "IPO/LTO enabled for Release build.")
  else()
    message(WARNING "IPO/LTO requested but not supported.")
  endif()
else()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
endif()

# Interface target to hold global compile options
add_library(numbits_compile_options INTERFACE)
target_compile_options(
  numbits_compile_options
  INTERFACE -Wall -Wextra $<$<CONFIG:Release>:-O3;-DNDEBUG>
            $<$<CONFIG:Debug>:-g>)

if(ENABLE_NATIVE)
  target_compile_options(
    numbits_compile_options
    INTERFACE
      $<$<AND:$<NOT:$<CONFIG:Debug>>,$<COMPILE_LANGUAGE:CXX>>:-march=native>)
endif()

if(ENABLE_FAST_MATH)
  target_compile_options(
    numbits_compile_options
    INTERFACE
      $<$<AND:$<NOT:$<CONFIG:Debug>>,$<COMPILE_LANGUAGE:CXX>>:-ffast-math>)
endif()

# Header-only interface library
add_library(numbits INTERFACE)
add_library(numbits::numbits ALIAS numbits)
target_include_directories(numbits INTERFACE ${PROJECT_SOURCE_DIR}/include/)
target_link_libraries(numbits INTERFACE numbits_compile_options)

# Optional OpenMP support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(numbits INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(numbits INTERFACE USE_OPENMP)
endif()

# Python bindings (nanobind)
if(BUILD_PYTHON)
  add_subdirectory(src)
endif()
