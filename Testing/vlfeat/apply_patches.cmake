# Applies patches to a source tree with 'git apply'. Run as a script:
#
#   cmake -DGIT_EXECUTABLE=<git> -DSOURCE_DIR=<dir> -DPATCHES=<a.patch|b.patch> -P apply_patches.cmake
#
# Patches are separated by '|'. A patch that is already applied is skipped, so
# running the script again on the same tree is safe. Works on every platform
# that has git, without a 'patch' tool or a shell.

foreach(_var GIT_EXECUTABLE SOURCE_DIR PATCHES)
  if(NOT DEFINED ${_var})
    message(FATAL_ERROR "apply_patches.cmake: ${_var} is not set")
  endif()
endforeach()

string(REPLACE "|" ";" PATCHES "${PATCHES}")

# Keep git from treating SOURCE_DIR as part of an enclosing repository, for
# example when the build tree lives inside the PopSift clone. Git then only
# uses SOURCE_DIR/.git if it exists, and otherwise patches the plain files.
get_filename_component(_parent "${SOURCE_DIR}" DIRECTORY)
set(ENV{GIT_CEILING_DIRECTORIES} "${_parent}")

foreach(_patch IN LISTS PATCHES)
  get_filename_component(_name "${_patch}" NAME)

  execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${_patch}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE _already_applied
    OUTPUT_QUIET ERROR_QUIET)
  if(_already_applied EQUAL 0)
    message(STATUS "${_name}: already applied, skipping")
    continue()
  endif()

  execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --whitespace=nowarn "${_patch}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE _result
    ERROR_VARIABLE _error)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "${_name}: git apply failed in ${SOURCE_DIR}:\n${_error}")
  endif()
  message(STATUS "${_name}: applied")
endforeach()
