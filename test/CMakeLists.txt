include(FetchContent)
# include(CTest)

#
# Consider replacing FetchContent with file(DOWNLOAD ...)
# Docs: https://cmake.org/cmake/help/latest/command/file.html#download
#
# or perhaps with ExternalData_Add_Test
# Docs: https://cmake.org/cmake/help/latest/module/ExternalData.html#module:ExternalData
#

# There is no need to check whether dataset files have been updates online. The haven't.
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

set_property(GLOBAL PROPERTY PopSift_TESTFILES_DIRS "")

function(download_test_content name url hash)
	# Declare the URL to fetch the dataset file, its hash, and ask for storage in the source tree
	FetchContent_Declare(
		${name}
  		URL      ${url}
  		URL_HASH MD5=${hash}
		SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dataset
		DOWNLOAD_NO_EXTRACT true
	)

	# Do the actual download
	FetchContent_MakeAvailable(${name})

	# Fill the variables _POPULATED, _SOURCE_DIR and _BINARY_DIR
	FetchContent_GetProperties(${name})

	if(NOT ${name}_POPULATED)
		message(ERROR "${name} has not been downloaded")
	else()
		# Extract the testfile from the archive into the build directory
		set(dir "${CMAKE_BINARY_DIR}/dataset/${name}")

		get_property(tmp GLOBAL PROPERTY PopSift_TESTFILES_DIRS)

		file(ARCHIVE_EXTRACT INPUT ${${name}_SOURCE_DIR}/boat.tar.gz DESTINATION ${dir})
		file(GLOB images "${dir}/*.pgm")
		foreach(img "${images}")
			if("${tmp}" STREQUAL "")
				set(tmp "${img}")
			else()
				set(tmp "${tmp};${img}")
			endif()
		endforeach()

		set_property(GLOBAL PROPERTY PopSift_TESTFILES_DIRS "${tmp}")
	endif()
endfunction(download_test_content)

# The files of the Oxford dataset: boat bikes trees graf wall bark leuven ubc
download_test_content(oxford_boat   "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/boat.tar.gz"   "d0d4c590adf5ab9a4f7aa39955022f7c")
download_test_content(oxford_bikes  "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/bikes.tar.gz"  "2809210051e160009c972595dfa42166")
download_test_content(oxford_trees  "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/trees.tar.gz"  "359991610db765c5ea42197734d2fa1f")
download_test_content(oxford_graf   "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graf.tar.gz"   "5fba3852d4c0855f8756ad8e1998efed")
download_test_content(oxford_wall   "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/wall.tar.gz"   "6b7b381313864458580b6c580bfbf30b")
download_test_content(oxford_bark   "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/bark.tar.gz"   "83718e945463cdbe5ca687af125331af")
download_test_content(oxford_leuven "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/leuven.tar.gz" "b0fe0574b1fb14599a44eedeb087fb8d")
download_test_content(oxford_ubc    "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/ubc.tar.gz"    "3ff19120715b59ed8ea1bf5346ba99c0")

get_property(tmp GLOBAL PROPERTY PopSift_TESTFILES_DIRS)

if(PopSift_TESTFILES_DIRS_VERBOSE)
  foreach(img ${tmp})
    message(STATUS "Image: ${img}")
  endforeach()
endif()

if(PopSift_USE_TESTS)
  message(STATUS "Testing popsift-demo for ${tmp}")

  foreach(img ${tmp})
    cmake_path(GET img STEM FName)
    cmake_path(GET img PARENT_PATH DName)
    cmake_path(GET DName STEM PName)
    set(TName "${PName}-${FName}")

    add_test(NAME ${TName} COMMAND popsift-demo -i ${img})
  endforeach()
endif()

