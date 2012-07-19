cmake_minimum_required (VERSION 2.8 FATAL_ERROR)

project(HMM)

##-- version stuff
set(HMM_VERSION_CURRENT "0")
set(HMM_VERSION_REVISION "0")
set(HMM_VERSION_AGE "1")
set(HMM_VERSION "${HMM_VERSION_CURRENT}.${HMM_VERSION_REVISION}.${HMM_VERSION_AGE}")

##-- options
option(BUILD_EXAMPLES "Build HMM usage examples (see Examples subdirectory)" ON)
option(USE_OPENMP "Use OpenMP parallelization" ON)
option(BUILD_DOC "Use Doxygen to build API documentation" OFF)

if(UNIX AND NOT WIN32)
  #-- set bin
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  #-- build static or shared library type ?
  option(BUILD_HMM_SHARED "Build HMM as shared library" ON)
  if(BUILD_HMM_SHARED)
    set(HMM_LIBRARY_TYPE SHARED)
  else(BUILD_HMM_SHARED)
    set(HMM_LIBRARY_TYPE STATIC)
  endif(BUILD_HMM_SHARED)

  #-- doxygen
  if(BUILD_DOC)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
      configure_file(Doxyfile.in ${PROJECT_BINARY_DIR}/Doxyfile @ONLY IMMEDIATE)
      add_custom_target(doc ALL
	COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/Doxyfile
	SOURCES ${PROJECT_BINARY_DIR}/Doxyfile)
    else(DOXYGEN_FOUND)
      message(FATAL_ERROR "Doxygen is needed to build API documentation. Please install it correctly.")
    endif(DOXYGEN_FOUND)
  endif(BUILD_DOC)

  #-- Boost
  find_package(Boost 1.42)
  if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    link_libraries(${Boost_LIBRARIES})
  else(Boost_FOUND)
    message("FATAL: Boost C++ libraries not found. Get Boost from boost.org or point BOOST_ROOT to your installation directory.")
  endif(Boost_FOUND)

  #-- OpenMP
  if(USE_OPENMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif(OPENMP_FOUND)
  endif(USE_OPENMP)

  #-- configure a header file to pass some of the CMake settings to the source code
  configure_file (
    "${PROJECT_SOURCE_DIR}/include/HMM_config.h.in"
    "${PROJECT_BINARY_DIR}/HMM_config.h"
    )

  #-- add the binary tree to the search path for include files
  include_directories("${PROJECT_SOURCE_DIR}/include") # our header files
  include_directories("${PROJECT_BINARY_DIR}") # so that we will find the generated config header

  #-- add HMM library
  file(
    GLOB_RECURSE
    source_files
    "${PROJECT_SOURCE_DIR}/src/*.cxx"
    )
  add_library(HMM ${HMM_LIBRARY_TYPE} ${source_files})

  #-- add Examples target
  if(BUILD_EXAMPLES)
    add_subdirectory(examples)
  endif(BUILD_EXAMPLES)

  message("
   
    ====================================================================================
    This program is free software: you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public License
    as published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this program. If not, see
    <http://www.gnu.org/licenses/>.

    HMM ${HMM_VERSION}
    Copyright (c) 2012 dohmatob elvis dopgima
    ====================================================================================
    ")
else (UNIX AND NOT WIN32)
  message("NOT FOR WIN32!")
endif(UNIX AND NOT WIN32)

