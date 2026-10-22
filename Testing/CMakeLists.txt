# The VLFeat comparison is a CTest, so it only exists when testing is enabled.
# This also keeps VLFeat from being downloaded in builds that do not test.
if(PopSift_TEST_WITH_VLFEAT AND BUILD_TESTING)
  set(_popsift_compare_vlfeat ON)
else()
  set(_popsift_compare_vlfeat OFF)
  if(PopSift_TEST_WITH_VLFEAT)
    message(STATUS "PopSift_TEST_WITH_VLFEAT is ON but BUILD_TESTING is OFF: "
                   "the VLFeat comparison is not configured.")
  endif()
endif()

# Which image sets to test with, where they are, and whether they are
# present. See datasets/README.md. Declares its options in every
# configuration and does the rest only when test images are needed.
include(datasets/TestImageSets.cmake)

if(_popsift_compare_vlfeat)
  # Fetch, patch and build the VLFeat reference (target vlfeat_sift)
  add_subdirectory(vlfeat)
endif()
