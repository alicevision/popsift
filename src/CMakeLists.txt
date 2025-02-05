# Do not specify SHARED or STATIC in add_library. Let the variable BUILD_SHARED_LIBS determine this.

add_library(popsift
        popsift/popsift.cc popsift/popsift.h
        popsift/features.cc popsift/features.h
        popsift/sift_constants.cc popsift/sift_constants.h
        popsift/sift_conf.cc popsift/sift_conf.h
        popsift/gauss_filter.cc popsift/gauss_filter.h
        popsift/s_image.cc popsift/s_image.h
        popsift/sift_pyramid.cc popsift/sift_pyramid.h
        popsift/sift_octave.cc popsift/sift_octave.h
        popsift/s_pyramid_build.cc
        popsift/s_pyramid_build_aa.cc
        popsift/s_pyramid_build_ra.cc
        popsift/sift_extremum.h
        popsift/sift_extremum.cc popsift/s_extrema.cc
        popsift/s_orientation.cc
        popsift/s_filtergrid.cc
        popsift/sift_desc.cc
        popsift/s_desc_vlfeat.cc popsift/s_desc_vlfeat.h
        popsift/s_desc_norm_rs.h
        popsift/s_desc_norm_l2.h
        popsift/s_desc_normalize.h
        popsift/s_gradiant.h
        popsift/s_solve.h
        popsift/common/assist.cc popsift/common/assist.h
        popsift/common/clamp.h
        popsift/common/plane_base.cc popsift/common/plane_base.h
        popsift/common/plane_2d.h
        popsift/common/write_plane_2d.cc popsift/common/write_plane_2d.h
        popsift/common/debug_macros.cc popsift/common/debug_macros.h
        popsift/common/warp_bitonic_sort.h
        popsift/common/excl_blk_prefix_sum.h
        popsift/common/vec_macros.h
        popsift/common/clamp.h)

target_link_libraries(popsift
                      PUBLIC
		      Threads::Threads)

set_target_properties(popsift PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(popsift PROPERTIES DEBUG_POSTFIX "d")

# build directory containing the automatically generated files
set(popsift_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

# BUILD_INTERFACE allows to include the directory with source only when target is
# built in the building tree (ie, not from an install location)
target_include_directories(popsift
            PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<BUILD_INTERFACE:${popsift_generated_dir}>
            $<BUILD_INTERFACE:${popsift_generated_dir}/popsift>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
	    )

# EXPORTING THE LIBRARY
#
# place to put the cmake-related files
set(popsift_config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
# include directory for install
set(popsift_include_install_dir "${CMAKE_INSTALL_INCLUDEDIR}")
# the name for the generated header version file
set(popsift_version_header_name "${popsift_generated_dir}/popsift/version.hpp")
# the name for the generated config.hpp
set(popsift_config_header_name "${popsift_generated_dir}/popsift/sift_config.h")
# where to install the generated files
set(popsift_install_dir_generated "${popsift_include_install_dir}/popsift")


# Configuration
set(popsift_cmake_version_config "${popsift_generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(popsift_cmake_project_config "${popsift_generated_dir}/${PROJECT_NAME}Config.cmake")
set(popsift_targets_export_name "${PROJECT_NAME}Targets")
set(popsift_namespace "${PROJECT_NAME}::")

# Include module with fuction 'write_basic_package_version_file'
include(CMakePackageConfigHelpers)

# Configure '<PROJECT-NAME>ConfigVersion.cmake'
# Note: major version number must be the same as requested
write_basic_package_version_file("${popsift_cmake_version_config}" COMPATIBILITY SameMajorVersion)

# Configure '<PROJECT-NAME>Config.cmake'
# Use variables:
#   * popsift_targets_export_name
#   * PROJECT_NAME
configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in"
                              "${popsift_cmake_project_config}"
                              INSTALL_DESTINATION "${popsift_config_install_dir}")

# version file
configure_file("${PROJECT_SOURCE_DIR}/cmake/version.hpp.in" ${popsift_version_header_name} @ONLY)
# config file
configure_file("${PROJECT_SOURCE_DIR}/cmake/sift_config.h.in" ${popsift_config_header_name} @ONLY)


# Targets:
#   * <prefix>/lib/libpopsift.a
#   * header location after install: <prefix>/include/
#   * headers can be included by C++ code `#include <popsift/popsift.h>`
install(TARGETS popsift
        EXPORT "${popsift_targets_export_name}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${popsift_include_install_dir}")

# Headers:
install(DIRECTORY "popsift"
        DESTINATION "${popsift_include_install_dir}"
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")

install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/popsift"
        DESTINATION "${popsift_include_install_dir}"
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")

# Config
#   * <prefix>/lib/cmake/${PROJECT_NAME}/${PROJECT_NAME}Config.cmake
#   * <prefix>/lib/cmake/${PROJECT_NAME}${PROJECT_NAME}ConfigVersion.cmake
install(FILES "${popsift_cmake_project_config}" "${popsift_cmake_version_config}"
        DESTINATION "${popsift_config_install_dir}")

# Config
#   * <prefix>/lib/cmake/${PROJECT_NAME}/${PROJECT_NAME}Targets.cmake
install(EXPORT "${popsift_targets_export_name}"
        NAMESPACE "${popsift_namespace}"
        DESTINATION "${popsift_config_install_dir}")

# install the config and version generated files
install( FILES "${popsift_config_header_name}"
		DESTINATION "${popsift_install_dir_generated}")

install( FILES "${popsift_version_header_name}"
		DESTINATION "${popsift_install_dir_generated}")

# Generate ${PROJECT_NAME}Targets.cmake in the build directory to use the library without installing it
export(TARGETS popsift FILE "${popsift_generated_dir}/${popsift_targets_export_name}.cmake")

if(PopSift_BUILD_EXAMPLES)
  add_subdirectory(application)
endif()

