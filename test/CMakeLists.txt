# include(FetchContent)
include(CTest)

#
# Consider replacing FetchContent with file(DOWNLOAD ...)
# Docs: https://cmake.org/cmake/help/latest/command/file.html#download
#
# or perhaps with ExternalData_Add_Test
# Docs: https://cmake.org/cmake/help/latest/module/ExternalData.html#module:ExternalData
#

# set_property(GLOBAL PROPERTY PopSift_TESTFILES_DIRS "")

message(STATUS "Running CMake file the test directory")

function(download_test_content name url hash)
	set(localdir "${CMAKE_CURRENT_SOURCE_DIR}/dataset/${name}")
	set(localarchive "${localdir}.tar.gz")

	# Declare the URL to fetch the dataset file, its hash, and ask for storage in the source tree
	if(EXISTS ${localarchive})
		message(STATUS "Test image archive ${name} exists")
	else()
		message(STATUS "Download ${url} for ${localarchive}")
		file(DOWNLOAD ${url} ${localarchive} EXPECTED_HASH MD5=${hash} SHOW_PROGRESS STATUS status)
		list(GET status 0 errcode)
		list(GET status 1 errstring)
		if(${errcode})
			message(STATUS "Errror code is ${errcode}, error is ${errstring}")
			return()
		endif()
	endif()

	file(ARCHIVE_EXTRACT INPUT ${localarchive} DESTINATION ${localdir})

	get_property(tmp GLOBAL PROPERTY PopSift_TESTFILES_DIRS)
	list(APPEND tmp "${localdir}")
	set_property(GLOBAL PROPERTY PopSift_TESTFILES_DIRS "${tmp}")
endfunction(download_test_content)

# The files of the Oxford dataset: boat bikes trees graf wall bark leuven ubc
download_test_content(ubc    "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/ubc.tar.gz"    "3ff19120715b59ed8ea1bf5346ba99c0")
download_test_content(bark   "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/bark.tar.gz"   "83718e945463cdbe5ca687af125331af")
# download_test_content(boat   "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/boat.tar.gz"   "d0d4c590adf5ab9a4f7aa39955022f7c")
# download_test_content(graf   "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graf.tar.gz"   "5fba3852d4c0855f8756ad8e1998efed")
# download_test_content(wall   "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/wall.tar.gz"   "6b7b381313864458580b6c580bfbf30b")
# download_test_content(bikes  "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/bikes.tar.gz"  "2809210051e160009c972595dfa42166")
# download_test_content(trees  "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/trees.tar.gz"  "359991610db765c5ea42197734d2fa1f")
# download_test_content(leuven "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/leuven.tar.gz" "b0fe0574b1fb14599a44eedeb087fb8d")

get_property(tmp GLOBAL PROPERTY PopSift_TESTFILES_DIRS)

# set(PopSift_TESTFILES_DIRS_VERBOSE OFF)

# if(PopSift_TESTFILES_DIRS_VERBOSE)
foreach(dir ${tmp})
message(STATUS "Test directory: ${dir}")
endforeach()
# endif()

# if(PopSift_USE_TESTS)
# message(STATUS "Testing popsift-demo for ${tmp}")

# message(STATUS "Adding test >>>compareOutput.py popsift-demo ${tmp}<<<")
# add_test(NAME compareOutput.py COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/compareOutput.py ${CMAKE_BINARY_DIR}/Linux-x86_64/popsift-demo ${tmp})
# endif()

