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

add_executable(popsift-demo main.cpp pgmread.cpp)
set_property(TARGET popsift-demo PROPERTY CXX_STANDARD 11)

target_include_directories(popsift-demo PUBLIC ${Boost_INCLUDE_DIRS})
target_compile_definitions(popsift-demo PRIVATE BOOST_ALL_NO_LIB BOOST_ALL_DYN_LINK)
target_link_libraries(popsift-demo PUBLIC PopSift::popsift ${Boost_LIBRARIES})

# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set_target_properties(popsift-demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )

install(TARGETS popsift-demo DESTINATION bin)
