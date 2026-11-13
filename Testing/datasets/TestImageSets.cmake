# ==============================================================================
# Test image sets
# ==============================================================================
#
# Decides which image sets the comparison scripts use. Nothing is ever
# downloaded; see download_oxford.py for the Oxford sets.
#
# Sets:
#   handcrafted   Testing/testImages, in the repository. Always used.
#   <path>        every folder below PopSift_TESTDATA_DIR that holds .pgm or
#                 .ppm files, named by its path relative to it, e.g.
#                 oxford/boat for <dir>/oxford/boat/.
#
# Included by Testing/CMakeLists.txt when BUILD_TESTING is ON. Afterwards
# these variables describe the sets:
#   PopSift_TEST_IMAGE_SETS              names, e.g. handcrafted;oxford/boat
#   PopSift_TEST_IMAGE_SET_<id>_DIR      the folder of the set
#   PopSift_TEST_IMAGE_SET_<id>_IMAGES   its images, relative to the folder
# where <id> is string(MAKE_C_IDENTIFIER) of the name.

set(PopSift_TESTDATA_DIR "$ENV{POPSIFT_TESTDATA_DIR}" CACHE PATH
    "Folder of test images: every folder below it that holds .pgm/.ppm files becomes a test image set. Default: $POPSIFT_TESTDATA_DIR.")

set(PopSift_TEST_IMAGE_SETS "")
set(_popsift_set_report "")

# Registers the images directly in a folder as the set ${name}. Adding or
# removing images re-runs configure at the next build.
function(_popsift_add_set name dir)
  file(GLOB _imgs RELATIVE "${dir}" CONFIGURE_DEPENDS
       "${dir}/*.pgm" "${dir}/*.ppm" "${dir}/*.PGM" "${dir}/*.PPM")
  list(SORT _imgs)
  list(LENGTH _imgs _count)
  string(MAKE_C_IDENTIFIER "${name}" _id)
  set(PopSift_TEST_IMAGE_SET_${_id}_DIR "${dir}" PARENT_SCOPE)
  set(PopSift_TEST_IMAGE_SET_${_id}_IMAGES "${_imgs}" PARENT_SCOPE)
  set(PopSift_TEST_IMAGE_SETS ${PopSift_TEST_IMAGE_SETS} "${name}" PARENT_SCOPE)
  set(_popsift_set_report ${_popsift_set_report}
      "  ${name}: ${_count} images in ${dir}" PARENT_SCOPE)
endfunction()

get_filename_component(_dir "${CMAKE_CURRENT_LIST_DIR}/../testImages" ABSOLUTE)
_popsift_add_set(handcrafted "${_dir}")

if(NOT PopSift_TESTDATA_DIR STREQUAL "")
  get_filename_component(_root "${PopSift_TESTDATA_DIR}" ABSOLUTE
                         BASE_DIR "${CMAKE_BINARY_DIR}")
  if(NOT IS_DIRECTORY "${_root}")
    message(FATAL_ERROR "PopSift_TESTDATA_DIR (${_root}) is not a folder.")
  endif()
  # Every folder with at least one image, found through its images.
  # CONFIGURE_DEPENDS also notices new folders.
  file(GLOB_RECURSE _all RELATIVE "${_root}" CONFIGURE_DEPENDS
       "${_root}/*.pgm" "${_root}/*.ppm" "${_root}/*.PGM" "${_root}/*.PPM")
  set(_folders "")
  foreach(_img IN LISTS _all)
    get_filename_component(_folder "${_img}" DIRECTORY)
    if(_folder STREQUAL "")
      set(_folder ".")  # an empty string would vanish from the list
    endif()
    list(APPEND _folders "${_folder}")
  endforeach()
  list(REMOVE_DUPLICATES _folders)
  list(SORT _folders)
  foreach(_folder IN LISTS _folders)
    if(_folder STREQUAL ".")
      # Images directly in PopSift_TESTDATA_DIR: the set is named after it.
      get_filename_component(_name "${_root}" NAME)
      _popsift_add_set("${_name}" "${_root}")
    else()
      _popsift_add_set("${_folder}" "${_root}/${_folder}")
    endif()
  endforeach()
  if(NOT _folders)
    message(WARNING "PopSift_TESTDATA_DIR (${_root}) holds no .pgm/.ppm files.")
  endif()
endif()

message(STATUS "Test image sets:")
foreach(_line IN LISTS _popsift_set_report)
  message(STATUS "${_line}")
endforeach()
