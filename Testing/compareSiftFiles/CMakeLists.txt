if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  # I am top-level project, i.e. I am not being include by another project
  cmake_minimum_required(VERSION 3.12)
  project(PopSiftCompareDescriptors LANGUAGES CXX)

  include(GNUInstallDirs)

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(PopSift_BOOST_USE_STATIC_LIBS)
  set(Boost_USE_STATIC_LIBS ON)
endif()
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
  # Prefer Boost's own cmake config.
  cmake_policy(SET CMP0167 NEW)
endif()
find_package(Boost 1.71.0 REQUIRED COMPONENTS filesystem program_options system)
if(WIN32)
  add_definitions("-DBOOST_ALL_NO_LIB")
endif(WIN32)

#############################################################
# compareSiftFiles
#############################################################

add_executable(popsift-compareSiftFiles
               compareSiftFiles.cpp
               csf_feat.cpp csf_feat.h
               csf_match.cpp csf_match.h
	       csf_options.cpp csf_options.h)

target_include_directories(popsift-compareSiftFiles PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(popsift-compareSiftFiles PUBLIC ${Boost_LIBRARIES})

# C++17 for <execution>
set_target_properties(popsift-compareSiftFiles PROPERTIES
                      CXX_STANDARD 17
                      CXX_STANDARD_REQUIRED ON)

# libstdc++ runs the parallel algorithms on TBB when its headers are
# installed, and then needs TBB at link time. Without TBB they run serially.
find_package(TBB QUIET CONFIG)
if(TBB_FOUND)
  target_link_libraries(popsift-compareSiftFiles PRIVATE TBB::tbb)
endif()

# SIMDification of small loops with gcc works with -O3 -march=native -ftree-vectorize
# How to make sure this happens?

#############################################################
# installation
#############################################################

install(TARGETS popsift-compareSiftFiles DESTINATION ${CMAKE_INSTALL_BINDIR})
