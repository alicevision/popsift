# ==============================================================================
# Test image sets
# ==============================================================================
#
# Decides which image sets the tests use, where they live and whether they are
# present. Configure never touches the network; downloads happen only through
# the popsift_testdata target, or through CTest when PopSift_TESTDATA_DOWNLOAD
# is ON.
#
# Sets:
#   handcrafted   Testing/testImages, in the repository. Always used.
#   <group>/<set> described by Testing/datasets/<group>/<set>.cmake and
#                 selected with PopSift_TEST_DATASETS. Lives in
#                 PopSift_TESTDATA_DIR/<group>/<set>/.
#   custom        any folder of .pgm/.ppm files, PopSift_TEST_IMAGES_OOT_DIR.
#
# Always include()d by Testing/CMakeLists.txt. Afterwards, if test images
# are in use, these variables describe the sets that tests can use:
#   PopSift_TEST_IMAGE_SETS                 names, e.g. handcrafted;oxford/boat
#   PopSift_TEST_IMAGE_SET_<id>_DIR         folder searched for images
#   PopSift_TEST_IMAGE_SET_<id>_FIXTURE     CTest fixture that downloads the
#                                           set, empty if none is needed
# where <id> is string(MAKE_C_IDENTIFIER) of the name.

set(PopSift_TEST_DATASETS "" CACHE STRING
    "Downloadable test image sets: a group (e.g. oxford) or group/set (e.g. oxford/boat;oxford/graf). See Testing/datasets/.")

if(DEFINED ENV{POPSIFT_TESTDATA_DIR} AND NOT "$ENV{POPSIFT_TESTDATA_DIR}" STREQUAL "")
  set(_popsift_testdata_default "$ENV{POPSIFT_TESTDATA_DIR}")
else()
  set(_popsift_testdata_default "${CMAKE_BINARY_DIR}/testdata")
endif()
set(PopSift_TESTDATA_DIR "${_popsift_testdata_default}" CACHE PATH
    "Where downloadable test image sets live, as <dir>/<group>/<set>/. Can point at an existing or shared copy. Default: $POPSIFT_TESTDATA_DIR, else <build>/testdata.")
option(PopSift_TESTDATA_DOWNLOAD
       "Let ctest download selected test image sets that are missing. Without it, only building the popsift_testdata target downloads anything." OFF)
set(PopSift_TEST_IMAGES_OOT_DIR "" CACHE PATH
    "Provide an out-of-tree image dir. Avoids downloading. Its .pgm/.ppm images become the test image set 'custom'.")

# The options above are always declared, so they show up in cmake-gui and
# ccmake. The rest only runs when something uses test images: the VLFeat
# comparison, or a request for sets on its own (which makes the
# popsift_testdata targets available).
if(NOT (_popsift_compare_vlfeat
        OR (BUILD_TESTING AND (PopSift_TEST_DATASETS OR PopSift_TEST_IMAGES_OOT_DIR OR PopSift_TESTDATA_DOWNLOAD))))
  if(PopSift_TEST_DATASETS OR PopSift_TEST_IMAGES_OOT_DIR OR PopSift_TESTDATA_DOWNLOAD)
    message(STATUS "Test image set options are ignored because BUILD_TESTING is OFF.")
  endif()
  return()
endif()

# PopSift_TESTFILE_PATH used to name the image folder. Its old default
# (testImages/) is ignored; any other value becomes the 'custom' set.
if(DEFINED PopSift_TESTFILE_PATH AND NOT PopSift_TESTFILE_PATH STREQUAL ""
   AND NOT PopSift_TESTFILE_PATH STREQUAL "testImages/")
  message(DEPRECATION "PopSift_TESTFILE_PATH is deprecated, use PopSift_TEST_IMAGES_OOT_DIR.")
  if(PopSift_TEST_IMAGES_OOT_DIR STREQUAL "")
    set(PopSift_TEST_IMAGES_OOT_DIR "${PopSift_TESTFILE_PATH}")
  endif()
endif()

set(_popsift_datasets_dir "${CMAKE_CURRENT_LIST_DIR}")
get_filename_component(PopSift_TESTDATA_DIR "${PopSift_TESTDATA_DIR}" ABSOLUTE
                       BASE_DIR "${CMAKE_BINARY_DIR}")

set(PopSift_TEST_IMAGE_SETS "")
set(_popsift_set_report "")

# Collects the .pgm/.ppm files below a folder into ${out}.
function(_popsift_glob_images dir out)
  file(GLOB_RECURSE _imgs "${dir}/*.pgm" "${dir}/*.ppm" "${dir}/*.PGM" "${dir}/*.PPM")
  set(${out} "${_imgs}" PARENT_SCOPE)
endfunction()

# Adds one line to the configure summary.
macro(_popsift_report name status detail)
  string(LENGTH "${name}" _len)
  set(_pad "")
  if(_len LESS 16)
    math(EXPR _n "16 - ${_len}")
    string(REPEAT " " ${_n} _pad)
  endif()
  list(APPEND _popsift_set_report "  ${name}${_pad}${status}  ${detail}")
endmacro()

# Registers a usable set.
macro(_popsift_add_set name dir fixture)
  string(MAKE_C_IDENTIFIER "${name}" _id)
  list(APPEND PopSift_TEST_IMAGE_SETS "${name}")
  set(PopSift_TEST_IMAGE_SET_${_id}_DIR "${dir}")
  set(PopSift_TEST_IMAGE_SET_${_id}_FIXTURE "${fixture}")
endmacro()

# Reads Testing/datasets/<name>.cmake into _ds_* variables.
function(_popsift_read_descriptor name)
  set(DATASET_URL "")
  set(DATASET_SHA256 "")
  set(DATASET_SIZE "")
  set(DATASET_IMAGES "")
  include("${_popsift_datasets_dir}/${name}.cmake")
  if(DATASET_URL STREQUAL "" OR DATASET_SHA256 STREQUAL "" OR NOT DATASET_IMAGES)
    message(FATAL_ERROR "Testing/datasets/${name}.cmake must set DATASET_URL, DATASET_SHA256 and DATASET_IMAGES.")
  endif()
  set(_ds_url    "${DATASET_URL}"    PARENT_SCOPE)
  set(_ds_sha256 "${DATASET_SHA256}" PARENT_SCOPE)
  set(_ds_size   "${DATASET_SIZE}"   PARENT_SCOPE)
  set(_ds_images "${DATASET_IMAGES}" PARENT_SCOPE)
endfunction()

# --- handcrafted --------------------------------------------------------------
set(_popsift_handcrafted_dir "${CMAKE_CURRENT_LIST_DIR}/../testImages")
get_filename_component(_popsift_handcrafted_dir "${_popsift_handcrafted_dir}" ABSOLUTE)
_popsift_glob_images("${_popsift_handcrafted_dir}" _imgs)
list(LENGTH _imgs _count)
_popsift_add_set(handcrafted "${_popsift_handcrafted_dir}" "")
_popsift_report(handcrafted "present" "${_count} images, Testing/testImages")

# --- downloadable sets --------------------------------------------------------
# Expand groups (a folder in Testing/datasets) into their sets.
set(_popsift_selected "")
foreach(_entry IN LISTS PopSift_TEST_DATASETS)
  if(IS_DIRECTORY "${_popsift_datasets_dir}/${_entry}")
    file(GLOB _descs RELATIVE "${_popsift_datasets_dir}" "${_popsift_datasets_dir}/${_entry}/*.cmake")
    list(SORT _descs)
    foreach(_d IN LISTS _descs)
      string(REGEX REPLACE "\\.cmake$" "" _d "${_d}")
      list(APPEND _popsift_selected "${_d}")
    endforeach()
  elseif(EXISTS "${_popsift_datasets_dir}/${_entry}.cmake")
    list(APPEND _popsift_selected "${_entry}")
  else()
    # Old style: a bare Oxford name such as 'boat'.
    file(GLOB _matches RELATIVE "${_popsift_datasets_dir}" "${_popsift_datasets_dir}/*/${_entry}.cmake")
    list(LENGTH _matches _n)
    if(_n EQUAL 1)
      string(REGEX REPLACE "\\.cmake$" "" _d "${_matches}")
      list(APPEND _popsift_selected "${_d}")
    else()
      file(GLOB _known RELATIVE "${_popsift_datasets_dir}" "${_popsift_datasets_dir}/*/*.cmake")
      string(REPLACE ".cmake" "" _known "${_known}")
      list(SORT _known)
      string(REPLACE ";" ", " _known "${_known}")
      message(FATAL_ERROR
        "PopSift_TEST_DATASETS: '${_entry}' is not a known test image set or group. "
        "Known sets: ${_known}")
    endif()
  endif()
endforeach()
list(REMOVE_DUPLICATES _popsift_selected)

if(_popsift_selected)
  add_custom_target(popsift_testdata)
endif()

foreach(_name IN LISTS _popsift_selected)
  _popsift_read_descriptor("${_name}")
  string(MAKE_C_IDENTIFIER "${_name}" _id)
  set(_dest "${PopSift_TESTDATA_DIR}/${_name}")
  string(REPLACE ";" "|" _images_arg "${_ds_images}")
  set(_download_cmd "${CMAKE_COMMAND}"
      "-DNAME=${_name}" "-DURL=${_ds_url}" "-DSHA256=${_ds_sha256}"
      "-DDEST=${_dest}" "-DIMAGES=${_images_arg}"
      -P "${_popsift_datasets_dir}/download_dataset.cmake")

  # Explicit download: cmake --build <build> --target popsift_testdata[_<id>]
  add_custom_target(popsift_testdata_${_id} COMMAND ${_download_cmd} VERBATIM)
  add_dependencies(popsift_testdata popsift_testdata_${_id})

  # Present means every listed image exists as .pgm or .ppm.
  set(_missing "")
  foreach(_img IN LISTS _ds_images)
    file(GLOB_RECURSE _found "${_dest}/${_img}.pgm" "${_dest}/${_img}.ppm")
    if(NOT _found)
      list(APPEND _missing "${_img}")
    endif()
  endforeach()

  if(_ds_size)
    if(_ds_size LESS 1048576)
      math(EXPR _kb "(${_ds_size} + 1023) / 1024")
      set(_size_text "${_kb} KB download")
    else()
      math(EXPR _mb "(${_ds_size} + 524288) / 1048576")
      set(_size_text "${_mb} MB download")
    endif()
  else()
    set(_size_text "download size unknown")
  endif()

  if(NOT _missing)
    _popsift_add_set("${_name}" "${_dest}" "")
    _popsift_report("${_name}" "present" "${_dest}")
  elseif(PopSift_TESTDATA_DOWNLOAD AND BUILD_TESTING)
    # CTest fetches it before the first test that uses it.
    add_test(NAME testdata/${_name} COMMAND ${_download_cmd})
    set_tests_properties(testdata/${_name} PROPERTIES
                         FIXTURES_SETUP TESTDATA_${_id}
                         LABELS "testdata;${_name}"
                         TIMEOUT 3600)
    _popsift_add_set("${_name}" "${_dest}" "TESTDATA_${_id}")
    _popsift_report("${_name}" "missing" "ctest downloads it (${_size_text})")
  else()
    _popsift_report("${_name}" "missing" "skipped: build target popsift_testdata, or set -DPopSift_TESTDATA_DOWNLOAD=ON (${_size_text})")
  endif()
endforeach()

# --- custom -------------------------------------------------------------------
if(NOT PopSift_TEST_IMAGES_OOT_DIR STREQUAL "")
  get_filename_component(_custom_dir "${PopSift_TEST_IMAGES_OOT_DIR}" ABSOLUTE
                         BASE_DIR "${CMAKE_SOURCE_DIR}")
  _popsift_glob_images("${_custom_dir}" _imgs)
  list(LENGTH _imgs _count)
  if(_count GREATER 0)
    _popsift_add_set(custom "${_custom_dir}" "")
    _popsift_report(custom "present" "${_count} images, ${_custom_dir}")
  else()
    _popsift_report(custom "missing" "skipped: no .pgm/.ppm files in ${_custom_dir}")
  endif()
endif()

message(STATUS "Test image sets (PopSift_TESTDATA_DIR = ${PopSift_TESTDATA_DIR}):")
foreach(_line IN LISTS _popsift_set_report)
  message(STATUS "${_line}")
endforeach()
