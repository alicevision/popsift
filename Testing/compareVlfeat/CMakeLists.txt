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
  \"gpu_probe\": \"$<TARGET_FILE:popsift-gpu-probe>\",
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
                 --target popsift-demo popsift-gpu-probe popsift-compareSiftFiles
                 --config "$<CONFIG>")
set_tests_properties(popsift_bins_build PROPERTIES FIXTURES_SETUP POPSIFT_BINS)

if(USE_HIP)
  set(_popsift_gpu_backend hip)
else()
  set(_popsift_gpu_backend cuda)
endif()

set(_popsift_compare "${Python3_EXECUTABLE}" "${_popsift_scripts_dir}/popsift_compare.py"
                     --tools "${_popsift_tools_json}")
set(_popsift_limits --max-avg-dist "${PopSift_VLFEAT_MAX_AVG_DESC_DIST}"
                    --min-recall "${PopSift_VLFEAT_MIN_RECALL}"
                    --min-precision "${PopSift_VLFEAT_MIN_PRECISION}"
                    --max-desc-dist "${PopSift_VLFEAT_MAX_DESC_DIST}"
                    --min-nn-agreement "${PopSift_VLFEAT_MIN_NN_AGREEMENT}")

# Every image of every set gets three tests, e.g. for oxford/boat/img1:
#   run-vlfeat/oxford/boat/img1    VLFeat on the image (label cpu)
#   run-popsift/oxford/boat/img1   PopSift with the parameter set
#                                  vlfeat-emulation (labels gpu and cuda or
#                                  hip), one at a time, skipped without a GPU
#   compare/oxford/boat/img1       the comparison of the two (label cpu),
#                                  skipped if the PopSift run was skipped
# and every set one report/<set> after its comparisons. All are labelled with
# the set name, so that e.g. 'ctest -L oxford/boat' runs one set. A set that
# is downloaded by CTest adds its download fixture to the runs.
set(_popsift_test_count 0)
foreach(_set IN LISTS PopSift_TEST_IMAGE_SETS)
  string(MAKE_C_IDENTIFIER "${_set}" _id)
  set(_download ${PopSift_TEST_IMAGE_SET_${_id}_FIXTURE})
  foreach(_img IN LISTS PopSift_TEST_IMAGE_SET_${_id}_IMAGES)
    string(REGEX REPLACE "\\.[^./]*$" "" _name "${_img}")
    string(MAKE_C_IDENTIFIER "${_set}/${_name}" _iid)
    set(_args --images "${_set}" --image "${_img}")

    add_test(NAME run-vlfeat/${_set}/${_name}
             COMMAND ${_popsift_compare} ${_args} --step run vlfeat)
    set_tests_properties(run-vlfeat/${_set}/${_name} PROPERTIES
                         FIXTURES_SETUP RUN_VLFEAT_${_iid}
                         FIXTURES_REQUIRED "VLFEAT;${_download}"
                         LABELS "cpu;vlfeat;${_set}")

    add_test(NAME run-popsift/${_set}/${_name}
             COMMAND ${_popsift_compare} ${_args} --step run vlfeat-emulation)
    set_tests_properties(run-popsift/${_set}/${_name} PROPERTIES
                         FIXTURES_SETUP RUN_POPSIFT_${_iid}
                         FIXTURES_REQUIRED "POPSIFT_BINS;${_download}"
                         RESOURCE_LOCK gpu
                         SKIP_RETURN_CODE 77
                         LABELS "gpu;${_popsift_gpu_backend};vlfeat;${_set}")

    add_test(NAME compare/${_set}/${_name}
             COMMAND ${_popsift_compare} ${_args} --step compare ${_popsift_limits}
                     vlfeat-emulation vlfeat)
    set_tests_properties(compare/${_set}/${_name} PROPERTIES
                         FIXTURES_REQUIRED "POPSIFT_BINS;RUN_VLFEAT_${_iid};RUN_POPSIFT_${_iid};REPORT_${_id}"
                         SKIP_RETURN_CODE 77
                         LABELS "cpu;vlfeat;${_set}")
    math(EXPR _popsift_test_count "${_popsift_test_count} + 3")
  endforeach()

  add_test(NAME report/${_set}
           COMMAND ${_popsift_compare} --images "${_set}" --step report
                   vlfeat-emulation vlfeat)
  set_tests_properties(report/${_set} PROPERTIES
                       FIXTURES_CLEANUP REPORT_${_id}
                       SKIP_RETURN_CODE 77
                       LABELS "cpu;vlfeat;${_set}")
  math(EXPR _popsift_test_count "${_popsift_test_count} + 1")
endforeach()

message(STATUS "VLFeat comparison: ${_popsift_test_count} tests, PopSift runs on "
               "${_popsift_gpu_backend} (labels gpu;${_popsift_gpu_backend})")
