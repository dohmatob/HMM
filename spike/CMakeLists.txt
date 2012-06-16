cmake_minimum_required (VERSION 2.6)

project(HMM)
set(EXECUTABLE_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

set(HMM_VERSION_CURRENT "0")
set(HMM_VERSION_REVISION "0")
set(HMM_VERSION_AGE "2")

if(UNIX AND NOT WIN32)
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

    Copyright (C) 2012 dohmatob elvis dopgima
    ====================================================================================
    ")

  # check Boost
  find_package(Boost 1.42)
  if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    link_libraries(${Boost_LIBRARIES})
    message("--  Boost include directories: ${Boost_INCLUDE_DIRS}")
    message("--  Boost library directories: ${Boost_LIBRARY_DIRS}")
    # message("--  Boost libraries: ${Boost_LIBRARIES}")
  else(Boost_FOUND)
    message("FATAL: Boost C++ libraries not found. Get Boost from boost.org or point BOOST_ROOT to your installation directory.")
  endif(Boost_FOUND)

  # configure a header file to pass some of the CMake settings
  # to the source code
  configure_file (
    "${PROJECT_SOURCE_DIR}/include/HMM_config.h.in"
    "${PROJECT_BINARY_DIR}/HMM_config.h"
    )
  
  # add the binary tree to the search path for include files
  include_directories("${PROJECT_SOURCE_DIR}/include") # our header files
  include_directories("${PROJECT_BINARY_DIR}") # so that we will find the generated config header

  # generate source files
  file(
    GLOB_RECURSE
    source_files
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    )

  add_executable(HMM ${source_files})
  
else (UNIX AND NOT WIN32)
  message("NOT FOR WIN32!")
endif(UNIX AND NOT WIN32)

