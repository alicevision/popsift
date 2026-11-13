# Tools and CTest tests for testing PopSift. Only added when BUILD_TESTING
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
  \"image_sets\": {
${_image_sets_json}
  }
}
"
     CONDITION "${_tools_json_condition}")

# --- CTest -------------------------------------------------------------------

if(PopSift_TEST_WITH_VLFEAT)
  # PopSift against VLFeat on every image of every set
  add_subdirectory(compareVlfeat)
endif()
