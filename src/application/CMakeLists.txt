if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  # I am top-level project, i.e. I am not being include by another project
  cmake_minimum_required(VERSION 3.12)
  project(PopsiftDemo LANGUAGES CXX)

  option(PopSift_BOOST_USE_STATIC_LIBS "Link examples with static Boost libraries" OFF)
  option(BUILD_SHARED_LIBS "Build shared libraries" ON)

  include(GNUInstallDirs)

  set(CMAKE_POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS})

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
endif()

# enable -fPIE for executables when -fpic
# https://cmake.org/cmake/help/v3.17/policy/CMP0083.html
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.14)
  cmake_policy(SET CMP0083 NEW)
  include(CheckPIESupported)
  check_pie_supported()
elseif(CMAKE_POSITION_INDEPENDENT_CODE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # manually add the link flag for gcc
  list(APPEND CMAKE_EXE_LINKER_FLAGS "-pie")
endif()

if(TARGET popsift)
  # when compiled in the repository the target is already defined
  add_library(PopSift::popsift ALIAS popsift)
else()
  # Add NO_CMAKE_BUILDS_PATH for windows if using CMake-GUI to build packages
  # to avoid searching in temporary build directory of Foo project
  # See 5:
  #    * http://www.cmake.org/cmake/help/v3.0/command/find_package.html
  find_package(PopSift CONFIG REQUIRED)
  message(STATUS "Found PopSift, version: ${PopSift_VERSION}")
endif()

find_package(DevIL COMPONENTS IL ILU) # yields IL_FOUND, IL_LIBRARIES, IL_INCLUDE_DIR

# for newer CMake versions and Boost 1.70 or newer must use Boost's make file
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.30)
  cmake_policy(SET CMP0167 NEW)
endif()
if(PopSift_BOOST_USE_STATIC_LIBS)
  set(Boost_USE_STATIC_LIBS ON)
endif()
find_package(Boost 1.71.0 REQUIRED COMPONENTS filesystem program_options system)
if(WIN32)
  add_definitions("-DBOOST_ALL_NO_LIB")
endif(WIN32)

set(PD_INCLUDE_DIRS    ${Boost_INCLUDE_DIRS})
set(PD_LINK_LIBS       ${Boost_LIBRARIES} ${CUDA_CUDADEVRT_LIBRARY})

if(IL_FOUND OR DevIL_FOUND)
  message(STATUS "DevIL found")
  set(PD_COMPILE_OPTIONS "-DUSE_DEVIL")
  list(APPEND PD_INCLUDE_DIRS ${IL_INCLUDE_DIR})
  list(APPEND PD_LINK_LIBS    ${IL_LIBRARIES} ${ILU_LIBRARIES})
else()
  message(WARNING "DevIL not found -- Falling back to pgmread")
  set(PD_COMPILE_OPTIONS "" )
endif()

#############################################################
# popsift-demo
#############################################################

add_executable(popsift-demo  main.cpp pgmread.cpp pgmread.h)

if(USE_HIP)
  # main.cpp/pgmread.cpp have NO device code; they consume popsift's host API only.
  # popsift now links the HIP device-compile interface (roc::rocthrust -> hip::device,
  # which injects "-x hip") PRIVATE and exposes only hip::host PUBLIC, so these
  # sources compile as plain C++ (no "-x hip") and just link the HIP runtime. They
  # still need the hip_compat <cuda_runtime.h> shim and the cuda_to_hip.h force-include
  # because popsift's public headers (device_prop.h) #include <cuda_runtime.h>; the
  # library keeps those PRIVATE, so the consumer adds them explicitly here. USE_HIP
  # selects the HIP branch of cuda_to_hip.h. C++17 for the compat header.
  set_property(TARGET popsift-demo PROPERTY CXX_STANDARD 17)
  target_compile_definitions(popsift-demo PRIVATE USE_HIP)
  target_include_directories(popsift-demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../popsift/hip_compat)
  target_compile_options(popsift-demo PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-include${CMAKE_CURRENT_SOURCE_DIR}/../popsift/cuda_to_hip.h>)
else()
  set_property(TARGET popsift-demo PROPERTY CXX_STANDARD 11)
endif()

target_compile_options(popsift-demo PRIVATE ${PD_COMPILE_OPTIONS} )
target_include_directories(popsift-demo PUBLIC PopSift::popsift ${PD_INCLUDE_DIRS})
target_compile_definitions(popsift-demo PRIVATE ${Boost_DEFINITIONS})
target_link_libraries(popsift-demo PUBLIC PopSift::popsift ${PD_LINK_LIBS})


#############################################################
# popsift-match
#############################################################

add_executable(popsift-match match.cpp pgmread.cpp pgmread.h)

if(USE_HIP)
  # See popsift-demo above: plain-C++ host consumer, no device code, no "-x hip".
  set_property(TARGET popsift-match PROPERTY CXX_STANDARD 17)
  target_compile_definitions(popsift-match PRIVATE USE_HIP)
  target_include_directories(popsift-match PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../popsift/hip_compat)
  target_compile_options(popsift-match PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-include${CMAKE_CURRENT_SOURCE_DIR}/../popsift/cuda_to_hip.h>)
else()
  set_property(TARGET popsift-match PROPERTY CXX_STANDARD 11)
endif()

target_compile_options(popsift-match PRIVATE ${PD_COMPILE_OPTIONS} )
target_include_directories(popsift-match PUBLIC PopSift::popsift ${PD_INCLUDE_DIRS})
target_compile_definitions(popsift-match PRIVATE ${Boost_DEFINITIONS})
target_link_libraries(popsift-match PUBLIC PopSift::popsift ${PD_LINK_LIBS})

#############################################################
# installation
#############################################################

install(TARGETS popsift-demo DESTINATION ${CMAKE_INSTALL_BINDIR})
