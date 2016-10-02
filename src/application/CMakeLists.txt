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

find_package(Boost 1.53.0 REQUIRED COMPONENTS filesystem)

add_executable(popsift-demo main.cpp pgmread.cpp)

target_include_directories(popsift-demo PUBLIC ${Boost_INCLUDE_DIRS})

target_link_libraries(popsift-demo PUBLIC PopSift::popsift ${Boost_LIBRARIES})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

install(TARGETS popsift-demo DESTINATION bin)