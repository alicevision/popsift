set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})

CUDA_INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

CUDA_ADD_LIBRARY(popsift STATIC
	popsift/popsift.cu popsift/popsift.h
	popsift/features.cu popsift/features.h
	popsift/sift_constants.cu popsift/sift_constants.h
	popsift/sift_conf.cu popsift/sift_conf.h
	popsift/gauss_filter.cu popsift/gauss_filter.h
	popsift/s_image.cu popsift/s_image.h
	popsift/sift_pyramid.cu popsift/sift_pyramid.h
	popsift/sift_octave.cu popsift/sift_octave.h
	popsift/s_pyramid_build.cu
	popsift/s_pyramid_build_aa.cu popsift/s_pyramid_build_aa.h
	popsift/s_pyramid_build_ai.cu popsift/s_pyramid_build_ai.h
	popsift/s_pyramid_build_ra.cu popsift/s_pyramid_build_ra.h
	popsift/s_pyramid_fixed.cu
	popsift/sift_extremum.h
	popsift/sift_extremum.cu popsift/s_extrema.cu
	popsift/s_orientation.cu
	popsift/sift_desc.cu
	popsift/s_desc_loop.cu popsift/s_desc_loop.h
	popsift/s_desc_iloop.cu popsift/s_desc_iloop.h
	popsift/s_desc_grid.cu popsift/s_desc_grid.h
	popsift/s_desc_igrid.cu popsift/s_desc_igrid.h
	popsift/s_desc_notile.cu popsift/s_desc_notile.h
	popsift/s_desc_norm_rs.h
	popsift/s_desc_norm_l2.h
	popsift/s_desc_normalize.h
	popsift/s_gradiant.h
	popsift/s_solve.h
	popsift/common/assist.cu popsift/common/assist.h
	popsift/common/clamp.h
	popsift/common/plane_2d.cu popsift/common/plane_2d.h
	popsift/common/write_plane_2d.cu popsift/common/write_plane_2d.h
	popsift/common/debug_macros.cpp popsift/common/debug_macros.h
	popsift/common/device_prop.cpp popsift/common/device_prop.h
	popsift/common/warp_bitonic_sort.h
	popsift/common/excl_blk_prefix_sum.h
	popsift/common/vec_macros.h
	popsift/common/clamp.h )

# BUILD_INTERFACE allows to include the directory with source only when target is
# built in the building tree (ie, not from an install location)
target_include_directories(popsift 
            PUBLIC ${Boost_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS}
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>")


set_target_properties(popsift PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(popsift PROPERTIES DEBUG_POSTFIX "d")

# cannot use PRIVATE here as there is a bug in FindCUDA and CUDA_ADD_LIBRARY
# https://gitlab.kitware.com/cmake/cmake/issues/16097
 target_link_libraries(popsift ${Boost_LIBRARIES} ${CUDA_CUDADEVRT_LIBRARY})


# EXPORTING THE LIBRARY
#
# place to put the cmake-related files
set(config_install_dir "lib/cmake/${PROJECT_NAME}")
# include directory for install
set(include_install_dir "include")

# build directory containing the generated files
set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Configuration
set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(targets_export_name "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

# Include module with fuction 'write_basic_package_version_file'
include(CMakePackageConfigHelpers)

# Configure '<PROJECT-NAME>ConfigVersion.cmake'
# Note: major version number must be the same as requested
write_basic_package_version_file("${version_config}" COMPATIBILITY SameMajorVersion)

# Configure '<PROJECT-NAME>Config.cmake'
# Use variables:
#   * targets_export_name
#   * PROJECT_NAME
configure_package_config_file("cmake/Config.cmake.in"
                              "${project_config}"
                              INSTALL_DESTINATION "${config_install_dir}")

# Targets:
#   * <prefix>/lib/libpopsift.a
#   * header location after install: <prefix>/include/
#   * headers can be included by C++ code `#include <popsift/popsift.h>`
install(TARGETS popsift
        EXPORT "${targets_export_name}"
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib"
        RUNTIME DESTINATION "bin"
        INCLUDES DESTINATION "${include_install_dir}")

# Headers:
install(DIRECTORY "popsift"
        DESTINATION "${include_install_dir}"
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")

# Config
#   * <prefix>/lib/cmake/${PROJECT_NAME}/${PROJECT_NAME}Config.cmake
#   * <prefix>/lib/cmake/${PROJECT_NAME}${PROJECT_NAME}ConfigVersion.cmake
install(FILES "${project_config}" "${version_config}"
        DESTINATION "${config_install_dir}")

# Config
#   * <prefix>/lib/cmake/${PROJECT_NAME}/${PROJECT_NAME}Targets.cmake
install(EXPORT "${targets_export_name}"
        NAMESPACE "${namespace}"
        DESTINATION "${config_install_dir}")

if(PopSift_BUILD_EXAMPLES)
  add_subdirectory(application)
endif()

