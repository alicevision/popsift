cmake_minimum_required(VERSION 3.0)
project(Popsift_CGT_PQT)

add_library(PopSift::popsift ALIAS popsift)

find_package(Boost 1.53.0 REQUIRED COMPONENTS program_options system filesystem)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
find_package(Gflags REQUIRED)

set(PD_INCLUDE_DIRS    ${Boost_INCLUDE_DIRS})
set(PD_LINK_LIBS       ${Boost_LIBRARIES} ${CUDA_CUDADEVRT_LIBRARY})

if(PopSift_USE_NVTX_PROFILING)
  list(APPEND PD_LINK_LIBS ${CUDA_NVTX_LIBRARY})
endif(PopSift_USE_NVTX_PROFILING)

#############################################################
# lib_pqt
#############################################################

CUDA_ADD_LIBRARY(pqt
	pqt/PerturbationProTree.cu
	pqt/bitonicSort.cu
	pqt/helper.cu
	pqt/ProductQuantization.cu
	pqt/ProQuantization.cu
	pqt/ProTree.cu
	pqt/VectorQuantization.cu)

#############################################################
# tool_createdb
#############################################################

add_executable(tool_createdb tool_createdb.cpp)

set_property(TARGET tool_createdb PROPERTY CXX_STANDARD 11)

target_compile_options(tool_createdb PRIVATE ${PD_COMPILE_OPTIONS} )
target_include_directories(tool_createdb PUBLIC ${PD_INCLUDE_DIRS})
target_compile_definitions(tool_createdb PRIVATE ${Boost_DEFINITIONS} BOOST_ALL_DYN_LINK BOOST_ALL_NO_LIB)
target_link_libraries(tool_createdb PUBLIC pqt PopSift::popsift ${PD_LINK_LIBS} ${GFLAGS_LIBRARIES})

set_target_properties(tool_createdb PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )

#############################################################
# tool_query
#############################################################

add_executable(tool_query tool_query.cpp)

set_property(TARGET tool_query PROPERTY CXX_STANDARD 11)

target_compile_options(tool_query PRIVATE ${PD_COMPILE_OPTIONS} )
target_include_directories(tool_query PUBLIC ${PD_INCLUDE_DIRS})
target_compile_definitions(tool_query PRIVATE ${Boost_DEFINITIONS} BOOST_ALL_DYN_LINK BOOST_ALL_NO_LIB)
target_link_libraries(tool_query PUBLIC pqt PopSift::popsift ${PD_LINK_LIBS} ${GFLAGS_LIBRARIES})

set_target_properties(tool_query  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )

#############################################################
# installation
#############################################################

install(TARGETS tool_query DESTINATION bin)
