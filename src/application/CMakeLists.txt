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

if(PopSift_BOOST_USE_STATIC_LIBS)
  set(Boost_USE_STATIC_LIBS ON)
endif()
find_package(Boost 1.53.0 REQUIRED COMPONENTS filesystem program_options system)
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

if(PopSift_USE_NVTX_PROFILING)
  list(APPEND PD_LINK_LIBS ${CUDA_NVTX_LIBRARY})
endif(PopSift_USE_NVTX_PROFILING)

#############################################################
# popsift-demo
#############################################################

add_executable(popsift-demo  main.cpp pgmread.cpp pgmread.h)

set_property(TARGET popsift-demo PROPERTY CXX_STANDARD 11)

target_compile_options(popsift-demo PRIVATE ${PD_COMPILE_OPTIONS} )
target_include_directories(popsift-demo PUBLIC ${PD_INCLUDE_DIRS})
target_compile_definitions(popsift-demo PRIVATE ${Boost_DEFINITIONS})
target_link_libraries(popsift-demo PUBLIC PopSift::popsift ${PD_LINK_LIBS})


#############################################################
# popsift-match
#############################################################

add_executable(popsift-match match.cpp pgmread.cpp pgmread.h)

set_property(TARGET popsift-match PROPERTY CXX_STANDARD 11)

target_compile_options(popsift-match PRIVATE ${PD_COMPILE_OPTIONS} )
target_include_directories(popsift-match PUBLIC ${PD_INCLUDE_DIRS})
target_compile_definitions(popsift-match PRIVATE ${Boost_DEFINITIONS})
target_link_libraries(popsift-match PUBLIC PopSift::popsift ${PD_LINK_LIBS})

#############################################################
# installation
#############################################################

install(TARGETS popsift-demo DESTINATION ${CMAKE_INSTALL_BINDIR})
