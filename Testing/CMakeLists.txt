# Tools and CTest for testing PopSift. Only added when PopSift_BUILD_TESTING
# is ON. See README.md in this folder.

add_subdirectory(compareSiftFiles)

if(PopSift_TEST_WITH_VLFEAT)
  # Fetch, patch and build the VLFeat reference (target vlfeat_sift)
  add_subdirectory(vlfeat)
endif()

# Tells the comparison scripts whether there is a usable GPU
add_subdirectory(gpuProbe)

# Which image sets the comparison scripts use (handcrafted and the folders
# below PopSift_TESTDATA_DIR). See datasets/README.md.
include(datasets/TestImageSets.cmake)

# --- popsift-testing-config.json ---------------------------------------------
#
# Tells the scripts in scripts/ where the binaries, the parameter sets and the
# image sets are. An empty path means that the binary is not built. A build
# directory has one such file: with a multi-config generator (Visual Studio,
# Xcode, Ninja Multi-Config) it holds the paths of the Debug configuration,
# which is easier to step through when a comparison points to a bug.

set(_image_sets_json "")
foreach(_set IN LISTS PopSift_TEST_IMAGE_SETS)
  string(MAKE_C_IDENTIFIER "${_set}" _id)
  if(NOT _image_sets_json STREQUAL "")
    string(APPEND _image_sets_json ",\n")
  endif()
  string(APPEND _image_sets_json "    \"${_set}\": \"${PopSift_TEST_IMAGE_SET_${_id}_DIR}\"")
endforeach()

if(TARGET popsift-demo)
  set(_popsift_demo_path "$<TARGET_FILE:popsift-demo>")
else()
  set(_popsift_demo_path "")
endif()
if(PopSift_TEST_WITH_VLFEAT)
  set(_vlfeat_sift_path "${PopSift_VLFEAT_SIFT_BIN}")
else()
  set(_vlfeat_sift_path "")
endif()

get_property(_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_multi_config)
  set(_tools_json_condition "$<CONFIG:Debug>")
else()
  set(_tools_json_condition 1)
endif()

# The build options that collect_run.py records in the run.json of a result
# store: the PopSift_* options and a few of CMake's own, without the paths
# and settings of the tests themselves.
set(_build_json "    \"version\": \"${PROJECT_VERSION}\",\n    \"config\": \"$<CONFIG>\"")
get_cmake_property(_cache_vars CACHE_VARIABLES)
foreach(_var IN LISTS _cache_vars)
  if(_var MATCHES "^(PopSift_|USE_HIP$|BUILD_SHARED_LIBS$|CMAKE_CUDA_ARCHITECTURES$|CMAKE_HIP_ARCHITECTURES$)"
     AND NOT _var MATCHES "_(SOURCE_DIR|BINARY_DIR|IS_TOP_LEVEL)$|^PopSift_TEST(DATA_DIR|_COMPARE_LIMITS)$|^PopSift_VLFEAT_")
    string(REPLACE "\\" "\\\\" _value "${${_var}}")
    string(REPLACE "\"" "\\\"" _value "${_value}")
    string(APPEND _build_json ",\n    \"${_var}\": \"${_value}\"")
  endif()
endforeach()

file(GENERATE
     OUTPUT "${CMAKE_BINARY_DIR}/popsift-testing-config.json"
     CONTENT
"{
  \"popsift_demo\": \"${_popsift_demo_path}\",
  \"compare_sift_files\": \"$<TARGET_FILE:popsift-compareSiftFiles>\",
  \"gpu_probe\": \"$<TARGET_FILE:popsift-gpu-probe>\",
  \"vlfeat_sift\": \"${_vlfeat_sift_path}\",
  \"paramsets\": \"${CMAKE_CURRENT_SOURCE_DIR}/paramsets.json\",
  \"work_dir\": \"${CMAKE_CURRENT_BINARY_DIR}/run\",
  \"stores_dir\": \"${CMAKE_CURRENT_BINARY_DIR}/stores\",
  \"build\": {
${_build_json}
  },
  \"image_sets\": {
${_image_sets_json}
  }
}
"
     CONDITION "${_tools_json_condition}")

# --- CTest -------------------------------------------------------------------

find_package(Python3 3.8 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  # The result stores of the comparison scripts
  add_test(NAME scripts/result_store
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_result_store.py")
  set_tests_properties(scripts/result_store PROPERTIES LABELS "cpu;unit")
endif()

if(PopSift_TEST_WITH_VLFEAT)
  # PopSift against VLFeat on every image of every set
  add_subdirectory(compareVlfeat)
endif()
