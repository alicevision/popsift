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
find_package(Boost 1.53.0 REQUIRED COMPONENTS filesystem program_options system)
if(WIN32)
  add_definitions("-DBOOST_ALL_NO_LIB")
endif(WIN32)

#############################################################
# compareSiftFiles
#############################################################

add_executable(popsift-compareSiftFiles
               compareSiftFiles.cpp
               csf_feat.cpp csf_feat.h
	       csf_options.cpp csf_options.h)

target_include_directories(popsift-compareSiftFiles PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(popsift-compareSiftFiles PUBLIC ${Boost_LIBRARIES})

set_property(TARGET popsift-compareSiftFiles PROPERTY CXX_STANDARD 14)

#############################################################
# installation
#############################################################

install(TARGETS popsift-compareSiftFiles DESTINATION ${CMAKE_INSTALL_BINDIR})
