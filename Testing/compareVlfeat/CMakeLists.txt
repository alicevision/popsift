# CTest comparison between PopSift and VLFeat

find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)

set(PopSift_VLFEAT_MAX_AVG_DESC_DIST "-1.0"
    CACHE STRING "ctest needs pass/fail indicators. Set a positive number of compare average descriptor distance. -1 disables and reports only.")

# Optional pass/fail limits on the matched keypoints of each image (see
# Testing/scripts/match_stats.py). Negative values disable a limit, so by
# default the tests only report.
set(PopSift_VLFEAT_MIN_RECALL "-1" CACHE STRING
    "Fail an image where PopSift matches fewer than this share of VLFeat's keypoints. Negative: off.")
set(PopSift_VLFEAT_MIN_PRECISION "-1" CACHE STRING
    "Fail an image where fewer than this share of PopSift's keypoints match one of VLFeat's. Negative: off.")
set(PopSift_VLFEAT_MAX_DESC_DIST "-1" CACHE STRING
    "Fail an image whose median descriptor distance of matched keypoints exceeds this. Negative: off.")
set(PopSift_VLFEAT_MIN_NN_AGREEMENT "-1" CACHE STRING
    "Fail an image where fewer than this share of matched keypoints have their match as descriptor nearest neighbour. Negative: off.")

set(_popsift_scripts_dir "${CMAKE_CURRENT_SOURCE_DIR}/../scripts")
get_filename_component(_popsift_scripts_dir "${_popsift_scripts_dir}" ABSOLUTE)
get_filename_component(_popsift_paramsets "${CMAKE_CURRENT_SOURCE_DIR}/../paramsets.json" ABSOLUTE)

# --- Where the scripts find binaries, parameter sets and image sets --------
#
# One file per configuration for multi-config generators
# (popsift-testing-<config>.json), popsift-testing.json otherwise.

set(_image_sets_json "")
foreach(_set IN LISTS PopSift_TEST_IMAGE_SETS)
  string(MAKE_C_IDENTIFIER "${_set}" _id)
  if(NOT _image_sets_json STREQUAL "")
    string(APPEND _image_sets_json ",\n")
  endif()
  string(APPEND _image_sets_json "    \"${_set}\": \"${PopSift_TEST_IMAGE_SET_${_id}_DIR}\"")
endforeach()

set(_popsift_tools_json "${CMAKE_BINARY_DIR}/popsift-testing$<$<BOOL:$<CONFIG>>:-$<CONFIG>>.json")
file(GENERATE OUTPUT "${_popsift_tools_json}" CONTENT
"{
  \"popsift_demo\": \"$<TARGET_FILE:popsift-demo>\",
  \"compare_sift_files\": \"$<TARGET_FILE:popsift-compareSiftFiles>\",
  \"vlfeat_sift\": \"${PopSift_VLFEAT_SIFT_BIN}\",
  \"paramsets\": \"${_popsift_paramsets}\",
  \"work_dir\": \"${CMAKE_CURRENT_BINARY_DIR}/run\",
  \"image_sets\": {
${_image_sets_json}
  }
}
")

# --- The comparison tests ----------------------------------------------------

# Compile popsift if needed. An empty --config is ignored by single-config
# generators.
add_test(NAME popsift_bins_build
         COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                 --target popsift-demo --target popsift-compareSiftFiles
                 --config "$<CONFIG>")
set_tests_properties(popsift_bins_build PROPERTIES FIXTURES_SETUP POPSIFT_BINS)

# One comparison per test image set, labelled with the set name, so that
# e.g. 'ctest -L oxford/boat' runs one set. A set that is downloaded by
# CTest adds its download fixture.
foreach(_set IN LISTS PopSift_TEST_IMAGE_SETS)
  string(MAKE_C_IDENTIFIER "${_set}" _id)
  set(_fixtures VLFEAT POPSIFT_BINS ${PopSift_TEST_IMAGE_SET_${_id}_FIXTURE})
  add_test(NAME vlfeat-comparison/${_set}
           COMMAND "${Python3_EXECUTABLE}" "${_popsift_scripts_dir}/popsift_compare.py"
                   --tools "${_popsift_tools_json}"
                   --images "${_set}"
                   --max-avg-dist "${PopSift_VLFEAT_MAX_AVG_DESC_DIST}"
                   --min-recall "${PopSift_VLFEAT_MIN_RECALL}"
                   --min-precision "${PopSift_VLFEAT_MIN_PRECISION}"
                   --max-desc-dist "${PopSift_VLFEAT_MAX_DESC_DIST}"
                   --min-nn-agreement "${PopSift_VLFEAT_MIN_NN_AGREEMENT}"
                   vlfeat-emulation vlfeat)
  set_tests_properties(vlfeat-comparison/${_set} PROPERTIES
                       FIXTURES_REQUIRED "${_fixtures}"
                       LABELS "gpu;vlfeat;${_set}")
endforeach()

