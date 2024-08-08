include(FetchContent)
# include(CTest)

# There is no need to check whether dataset files have been updates online. The haven't.
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

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
		file(ARCHIVE_EXTRACT INPUT ${${name}_SOURCE_DIR}/boat.tar.gz DESTINATION ${CMAKE_BINARY_DIR}/dataset/${name})

		# Store the testfile dir in the CMake cache
		SET(TESTFILES_${name} "${CMAKE_BINARY_DIR}/dataset/${name}" CACHE INTERNAL "TESTFILES_${name}")
	endif()
endfunction(download_test_content)

# boat bikes trees graf wall bark leuven ubc
download_test_content(oxford_boat   "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/boat.tar.gz"   "d0d4c590adf5ab9a4f7aa39955022f7c")
# download_test_content(oxford_bikes  "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/bikes.tar.gz"  "2809210051e160009c972595dfa42166")
# download_test_content(oxford_trees  "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/trees.tar.gz"  "359991610db765c5ea42197734d2fa1f")
# download_test_content(oxford_graf   "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graf.tar.gz"   "5fba3852d4c0855f8756ad8e1998efed")
# download_test_content(oxford_wall   "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/wall.tar.gz"   "6b7b381313864458580b6c580bfbf30b")
# download_test_content(oxford_bark   "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/bark.tar.gz"   "83718e945463cdbe5ca687af125331af")
# download_test_content(oxford_leuven "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/leuven.tar.gz" "b0fe0574b1fb14599a44eedeb087fb8d")
# download_test_content(oxford_ubc    "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/ubc.tar.gz"    "3ff19120715b59ed8ea1bf5346ba99c0")

message(STATUS "boat files are in ${TESTFILES_oxford_boat}")


# ./_deps/oxford_boat-subbuild/oxford_boat-populate-prefix/src/boat.tar.gz

