cmake_minimum_required(VERSION 3.0)
project(PopsiftDemo)

if(TARGET popsift)
  # when compiled in the repository the target is already defined
  add_library(PopSift::popsift ALIAS popsift)
else()
  # Add NO_CMAKE_BUILDS_PATH for windows if using CMake-GUI to build packages
  # to avoid searching in temporary build directory of Foo project
  # See 5:
  #    * http://www.cmake.org/cmake/help/v3.0/command/find_package.html
  find_package(PopSift CONFIG REQUIRED)
endif()

find_package(Boost 1.53.0 REQUIRED COMPONENTS program_options system filesystem)
find_package(DevIL COMPONENTS IL ILU) # yields IL_FOUND, IL_LIBRARIES, IL_INCLUDE_DIR

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
target_compile_definitions(popsift-demo PRIVATE ${Boost_DEFINITIONS} BOOST_ALL_DYN_LINK BOOST_ALL_NO_LIB)
target_link_libraries(popsift-demo PUBLIC PopSift::popsift ${PD_LINK_LIBS})

set_target_properties(popsift-demo  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )

#############################################################
# popsift-match
#############################################################

add_executable(popsift-match match.cpp pgmread.cpp pgmread.h)

set_property(TARGET popsift-match PROPERTY CXX_STANDARD 11)

target_compile_options(popsift-match PRIVATE ${PD_COMPILE_OPTIONS} )
target_include_directories(popsift-match PUBLIC ${PD_INCLUDE_DIRS})
target_compile_definitions(popsift-match PRIVATE ${Boost_DEFINITIONS} BOOST_ALL_DYN_LINK BOOST_ALL_NO_LIB)
target_link_libraries(popsift-match PUBLIC PopSift::popsift ${PD_LINK_LIBS})

set_target_properties(popsift-match  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )

#############################################################
# installation
#############################################################

install(TARGETS popsift-demo DESTINATION bin)
