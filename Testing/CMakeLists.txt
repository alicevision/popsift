if(PopSift_USE_TEST_CMD)
  add_subdirectory(testScripts)
endif()

if(PopSift_COMPARE_VLFEAT)
  # Compile comparison tool
  add_subdirectory(compareSiftFiles)

  # Create ctest checks
  add_subdirectory(compareVlfeat)
endif()

