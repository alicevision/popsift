# Tools for testing PopSift. Only added when BUILD_TESTING is ON.

add_subdirectory(compareSiftFiles)

if(PopSift_TEST_WITH_VLFEAT)
  # Fetch, patch and build the VLFeat reference (target vlfeat_sift)
  add_subdirectory(vlfeat)
endif()
