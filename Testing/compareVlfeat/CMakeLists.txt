# CTest tests that compare PopSift with VLFeat on every image of every test
# image set. Added by Testing/CMakeLists.txt when BUILD_TESTING and
# PopSift_TEST_WITH_VLFEAT are ON. See README.md in this folder.

find_package(Python3 3.8 COMPONENTS Interpreter)
if(NOT Python3_Interpreter_FOUND)
  message(STATUS "No Python 3.8 or newer: no VLFeat comparison tests")
  return()
endif()
if(NOT TARGET popsift-demo)
  message(STATUS "PopSift_BUILD_EXAMPLES is OFF: no VLFeat comparison tests")
  return()
endif()

set(PopSift_TEST_COMPARE_LIMITS "" CACHE STRING
    "Options of popsift_compare.py that make an image fail, e.g. --min-recall;0.95;--max-desc-dist;12. Empty: the comparison tests only report.")

# The tests read popsift-testing-config.json, which a multi-config generator
# writes for the Debug configuration only.
get_property(_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_multi_config)
  set(_configurations CONFIGURATIONS Debug)
else()
  set(_configurations "")
endif()

if(USE_HIP)
  set(_gpu_backend hip)
else()
  set(_gpu_backend cuda)
endif()

set(_compare "${Python3_EXECUTABLE}"
             "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/popsift_compare.py"
             --tools "${CMAKE_BINARY_DIR}/popsift-testing-config.json")
set(_paramset popsift-vlfeat)

# Every image of every set gets three tests, e.g. for oxford/boat/img1:
#   run-vlfeat/oxford/boat/img1    VLFeat on the image (label cpu)
#   run-popsift/oxford/boat/img1   PopSift with the parameter set
#                                  popsift-vlfeat (labels gpu and cuda or
#                                  hip), one at a time, skipped without a GPU
#   compare/oxford/boat/img1       the comparison of the two (label cpu),
#                                  skipped if the PopSift run was skipped
# and every set one report/<set> after its comparisons. All carry the set
# name as a label, so that e.g. 'ctest -L oxford/boat' tests one set.
set(_count 0)
foreach(_set IN LISTS PopSift_TEST_IMAGE_SETS)
  string(MAKE_C_IDENTIFIER "${_set}" _id)
  foreach(_img IN LISTS PopSift_TEST_IMAGE_SET_${_id}_IMAGES)
    string(REGEX REPLACE "\\.[^./]*$" "" _name "${_img}")
    string(MAKE_C_IDENTIFIER "${_set}/${_name}" _iid)
    set(_args --images "${_set}" --image "${_img}")

    add_test(NAME run-vlfeat/${_set}/${_name} ${_configurations}
             COMMAND ${_compare} ${_args} --step run vlfeat)
    set_tests_properties(run-vlfeat/${_set}/${_name} PROPERTIES
                         FIXTURES_SETUP RUN_VLFEAT_${_iid}
                         LABELS "cpu;vlfeat;${_set}")

    add_test(NAME run-popsift/${_set}/${_name} ${_configurations}
             COMMAND ${_compare} ${_args} --step run ${_paramset})
    set_tests_properties(run-popsift/${_set}/${_name} PROPERTIES
                         FIXTURES_SETUP RUN_POPSIFT_${_iid}
                         RESOURCE_LOCK gpu
                         SKIP_RETURN_CODE 77
                         LABELS "gpu;${_gpu_backend};vlfeat;${_set}")

    add_test(NAME compare/${_set}/${_name} ${_configurations}
             COMMAND ${_compare} ${_args} --step compare
                     ${PopSift_TEST_COMPARE_LIMITS} ${_paramset} vlfeat)
    set_tests_properties(compare/${_set}/${_name} PROPERTIES
                         FIXTURES_REQUIRED "RUN_VLFEAT_${_iid};RUN_POPSIFT_${_iid};REPORT_${_id}"
                         SKIP_RETURN_CODE 77
                         LABELS "cpu;vlfeat;${_set}")
    math(EXPR _count "${_count} + 3")
  endforeach()

  # FIXTURES_CLEANUP: runs after all comparisons of the set
  add_test(NAME report/${_set} ${_configurations}
           COMMAND ${_compare} --images "${_set}" --step report
                   ${_paramset} vlfeat)
  set_tests_properties(report/${_set} PROPERTIES
                       FIXTURES_CLEANUP REPORT_${_id}
                       SKIP_RETURN_CODE 77
                       LABELS "cpu;vlfeat;${_set}")
  math(EXPR _count "${_count} + 1")
endforeach()

message(STATUS "VLFeat comparison: ${_count} tests, PopSift runs on "
               "${_gpu_backend} (labels gpu;${_gpu_backend})")
