cmake_minimum_required(VERSION 3.24)

# This must be called before project() or cmake sets it to the oldest non-deprecated CC
# "all" and "all-major" work for Intel and perhaps for ARM with discrete GPUs, but not Tegra and Jetson.
project(PopSift VERSION 1.0.0 LANGUAGES CXX)

# Set build path as a folder named as the platform (linux, windows, darwin...) plus the processor type
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

option(PopSift_BUILD_EXAMPLES "Build PopSift applications."  ON)
option(PopSift_BUILD_DOCS "Build PopSift documentation."  OFF)
option(PopSift_ERRCHK_AFTER_KERNEL     "Synchronize and check error after every kernel." OFF)
option(PopSift_USE_POSITION_INDEPENDENT_CODE "Generate position independent code." ON)
option(PopSift_USE_GRID_FILTER "Switch off grid filtering to massively reduce compile time while debugging other things." ON)
option(PopSift_USE_TEST_CMD "Add testing step for functional verification" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

if(PopSift_USE_POSITION_INDEPENDENT_CODE AND NOT MSVC)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# set(CMAKE_BUILD_TYPE Debug)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Build type not set, building in Release configuration")
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
else()
  message(STATUS "Building in ${CMAKE_BUILD_TYPE} configuration")
endif()

# ensure the proper linker flags when building the static version on MSVC
if(MSVC AND NOT BUILD_SHARED_LIBS)
  foreach(config "DEBUG" "RELEASE" "MINSIZEREL" "RELWITHDEBINFO")
    string(REPLACE /MD /MT CMAKE_C_FLAGS_${config} "${CMAKE_C_FLAGS_${config}}")
    string(REPLACE /MD /MT CMAKE_CXX_FLAGS_${config} "${CMAKE_CXX_FLAGS_${config}}")
    message(STATUS "CMAKE_C_FLAGS_${config} ${CMAKE_C_FLAGS_${config}}")
    message(STATUS "CMAKE_CXX_FLAGS_${config} ${CMAKE_CXX_FLAGS_${config}}")
  endforeach()
endif()

# ==============================================================================
# GNUInstallDirs CMake module
# - Define GNU standard installation directories
# - Provides install directory variables as defined by the GNU Coding Standards.
# ==============================================================================
include(GNUInstallDirs)


if(BUILD_SHARED_LIBS)
  message(STATUS "BUILD_SHARED_LIBS ON")

  # Auto-build dll exports on Windows
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  message(STATUS "BUILD_SHARED_LIBS OFF")
endif()

# Require threads because of std::thread.
find_package(Threads REQUIRED)

set(PopSift_CXX_STANDARD 20)

if(NOT MSVC)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++${PopSift_CXX_STANDARD}")
endif()
set(CMAKE_CXX_STANDARD ${PopSift_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(NOT PopSift_USE_GRID_FILTER)
  message(STATUS "Disabling grid filter compilation")
  set(DISABLE_GRID_FILTER   1)
else()
  set(DISABLE_GRID_FILTER   0)
endif()

add_subdirectory(src)

if(PopSift_BUILD_DOCS)
  add_subdirectory(doc)
endif()

set(PopSift_TESTFILE_PATH "popsift-samples/datasets/sample/big_set/" CACHE STRING "Base directory where your test files are stored")
if(PopSift_USE_TEST_CMD)
  if(NOT IS_ABSOLUTE("${PopSift_TESTFILE_PATH}"))
    get_filename_component(PopSift_TESTFILES "${PopSift_TESTFILE_PATH}" ABSOLUTE)
    set(PopSift_TESTFILE_PATH "${PopSift_TESTFILES}")
  endif()

  add_subdirectory(testScripts)
endif()

########### Add uninstall target ###############
CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)
ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake")


######################################
# SUMMARY
######################################
message("\n")
message("******************************************")
message("Building configuration:\n")
message(STATUS "PopSift version: " ${PROJECT_VERSION})
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
message(STATUS "Build Shared libs: " ${BUILD_SHARED_LIBS})
message(STATUS "Build examples: " ${PopSift_BUILD_EXAMPLES})
message(STATUS "Build documentation: " ${PopSift_BUILD_DOCS})
message(STATUS "Generate position independent code: " ${CMAKE_POSITION_INDEPENDENT_CODE})
message(STATUS "Synchronize and check error after every kernel: " ${PopSift_ERRCHK_AFTER_KERNEL})
message(STATUS "Grid filtering: " ${PopSift_USE_GRID_FILTER})
message(STATUS "Install path: " ${CMAKE_INSTALL_PREFIX})
message(STATUS "Testing step: " ${PopSift_USE_TEST_CMD})

if(PopSift_USE_TEST_CMD)
  message(STATUS "Path for test input: " ${PopSift_TESTFILE_PATH})
endif()
message("\n******************************************")
message("\n")
