# ==============================================================================
# Test dataset configuration, shared by PopSift_USE_TEST_CMD and
# PopSift_COMPARE_VLFEAT
# ==============================================================================
#
# PopSift_TESTFILE_PATH is searched recursively for .pgm/.ppm test images by
# both testScripts/ and compareVlfeat/. It can either already point at a
# directory that contains images (e.g. a shared/global dataset location), or
# be populated automatically by setting PopSift_TEST_DATASETS below.

set(PopSift_TESTFILE_PATH "popsift-samples/datasets/sample/big_set/" CACHE STRING
    "Base directory (searched recursively) where your .pgm/.ppm test files are stored.")

set(PopSift_TEST_DATASETS "" CACHE STRING
    "Oxford affine-covariant datasets for testing: boat;bikes;trees;graf;wall;bark;leuven;ubc. Populates PopSift_TESTFILE_PATH.")

if(PopSift_USE_TEST_CMD OR PopSift_COMPARE_VLFEAT)

  # --- Resolve PopSift_TESTFILE_PATH to an absolute path ---------------------
  # PopSift_TESTFILE_PATH is a path relative to CMAKE_SOURCE_DIR or absolute
  if(NOT IS_ABSOLUTE("${PopSift_TESTFILE_PATH}"))
    get_filename_component(_popsift_testfile_path_abs "${PopSift_TESTFILE_PATH}"
                            ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    set(PopSift_TESTFILE_PATH "${_popsift_testfile_path_abs}")
    unset(_popsift_testfile_path_abs)
  endif()

  # --- Optionally download and unpack Oxford datasets -------------------
  if(PopSift_TEST_DATASETS)
    set(_oxford_known_datasets boat bikes trees graf wall bark leuven ubc)
    set(_oxford_base_url "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files")

    foreach(_ds ${PopSift_TEST_DATASETS})
      if(NOT _ds IN_LIST _oxford_known_datasets)
        message(FATAL_ERROR
          "PopSift_TEST_DATASETS: '${_ds}' is not a known Oxford dataset. "
          "Choose one or more of: ${_oxford_known_datasets}.")
      endif()
    endforeach()

    # PopSift_TEST_DATASETS asks for datasets to be provisioned, so it owns
    # PopSift_TESTFILE_PATH from here on - point it at a fresh location under
    # the build tree rather than writing generated/downloaded content into
    # whatever PopSift_TESTFILE_PATH happened to be set to before.
    set(PopSift_TESTFILE_PATH "${CMAKE_CURRENT_BINARY_DIR}/oxford-datasets")
    file(MAKE_DIRECTORY "${PopSift_TESTFILE_PATH}")

    foreach(_ds ${PopSift_TEST_DATASETS})
      set(_ds_dir "${PopSift_TESTFILE_PATH}/${_ds}")
      file(GLOB _ds_existing_images "${_ds_dir}/*.pgm" "${_ds_dir}/*.ppm")
      if(_ds_existing_images)
        message(STATUS "PopSift_TEST_DATASETS: '${_ds}' already present at ${_ds_dir}, skipping download")
      else()
        set(_ds_archive "${CMAKE_CURRENT_BINARY_DIR}/${_ds}.tar.gz")
        message(STATUS "PopSift_TEST_DATASETS: downloading ${_oxford_base_url}/${_ds}.tar.gz ...")
        file(DOWNLOAD "${_oxford_base_url}/${_ds}.tar.gz" "${_ds_archive}"
             STATUS _ds_status SHOW_PROGRESS)
        list(GET _ds_status 0 _ds_errcode)
        list(GET _ds_status 1 _ds_errmsg)
        if(NOT _ds_errcode EQUAL 0)
          file(REMOVE "${_ds_archive}")
          message(FATAL_ERROR
            "PopSift_TEST_DATASETS: failed to download '${_ds}' from "
            "${_oxford_base_url}/${_ds}.tar.gz (${_ds_errmsg}).")
        endif()
        file(MAKE_DIRECTORY "${_ds_dir}")
        file(ARCHIVE_EXTRACT INPUT "${_ds_archive}" DESTINATION "${_ds_dir}")
        file(REMOVE "${_ds_archive}")
        message(STATUS "PopSift_TEST_DATASETS: '${_ds}' unpacked to ${_ds_dir}")
      endif()
      unset(_ds_existing_images)
      unset(_ds_dir)
    endforeach()
  endif()

  message(STATUS "Path for test input: ${PopSift_TESTFILE_PATH}")

  # Fail early and not during ctest if there is no dataset
  if(NOT EXISTS "${PopSift_TESTFILE_PATH}")
    message(FATAL_ERROR
      "Testing enabled but PopSift_TESTFILE_PATH ('${PopSift_TESTFILE_PATH}') does not exist. "
      "Either set -DPopSift_TESTFILE_PATH=<path> to an existing directory "
      "(local or shared/global) that contains .pgm/.ppm test images, or set "
      "-DPopSift_TEST_DATASETS=<dataset[;dataset...]> (any of boat, bikes, "
      "trees, graf, wall, bark, leuven, ubc) to have one downloaded "
      "automatically.")
  endif()

  file(GLOB_RECURSE _popsift_test_images
       "${PopSift_TESTFILE_PATH}/*.pgm" "${PopSift_TESTFILE_PATH}/*.ppm")

  if(NOT _popsift_test_images)
    message(FATAL_ERROR
      "Testing enabled but .pgm or .ppm files were not found under PopSift_TESTFILE_PATH "
      "('${PopSift_TESTFILE_PATH}').")
  endif()

  list(LENGTH _popsift_test_images _popsift_test_image_count)
  message(STATUS "Found ${_popsift_test_image_count} .pgm/.ppm test image(s) at/below ${PopSift_TESTFILE_PATH}")
  unset(_popsift_test_images)
  unset(_popsift_test_image_count)

endif()

if(PopSift_USE_TEST_CMD)
  add_subdirectory(testScripts)
endif()

if(PopSift_COMPARE_VLFEAT)
  # Compile comparison tool
  add_subdirectory(compareSiftFiles)

  # Create ctest checks
  add_subdirectory(compareVlfeat)
endif()

