# Downloads and unpacks one test image set. Run as a script:
#
#   cmake -DNAME=<group/set> -DURL=<url> -DSHA256=<hash> -DDEST=<dir>
#         [-DIMAGES=<img1|img2|...>] -P download_dataset.cmake
#
# If DEST already holds all IMAGES (as .pgm or .ppm), for example a copy the
# user provided, nothing is downloaded and DEST is left untouched.
#
# The archive is checked against SHA256 before it is unpacked into DEST.
# A stamp file records the hash, so a set that was downloaded completely is
# never fetched again. An interrupted download leaves no stamp and starts
# over the next time.

foreach(_var NAME URL SHA256 DEST)
  if(NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
    message(FATAL_ERROR "download_dataset.cmake: ${_var} is not set")
  endif()
endforeach()

set(_stamp "${DEST}/.popsift-dataset-sha256")
if(EXISTS "${_stamp}")
  file(READ "${_stamp}" _stamp_hash)
  string(STRIP "${_stamp_hash}" _stamp_hash)
  if(_stamp_hash STREQUAL SHA256)
    message(STATUS "${NAME}: already downloaded to ${DEST}")
    return()
  endif()
endif()

if(IMAGES)
  string(REPLACE "|" ";" IMAGES "${IMAGES}")
  set(_all_present TRUE)
  foreach(_img IN LISTS IMAGES)
    file(GLOB_RECURSE _found "${DEST}/${_img}.pgm" "${DEST}/${_img}.ppm")
    if(NOT _found)
      set(_all_present FALSE)
      break()
    endif()
  endforeach()
  if(_all_present)
    message(STATUS "${NAME}: all images present in ${DEST}, nothing to download")
    return()
  endif()
endif()

# Never delete files this script did not put there.
file(GLOB _existing "${DEST}/*")
if(_existing AND NOT EXISTS "${_stamp}")
  message(FATAL_ERROR
    "${NAME}: ${DEST} exists but is incomplete, and was not created by a "
    "PopSift download. Complete or remove it, or point PopSift_TESTDATA_DIR "
    "somewhere else.")
endif()

get_filename_component(_parent "${DEST}" DIRECTORY)
get_filename_component(_archive_name "${URL}" NAME)
string(MAKE_C_IDENTIFIER "${NAME}" _id)
set(_archive "${_parent}/.${_id}-${_archive_name}.part")
set(_unpack  "${_parent}/.${_id}-unpack")
file(MAKE_DIRECTORY "${_parent}")
file(REMOVE "${_archive}")
file(REMOVE_RECURSE "${_unpack}")

message(STATUS "${NAME}: downloading ${URL}")
file(DOWNLOAD "${URL}" "${_archive}"
     INACTIVITY_TIMEOUT 60
     SHOW_PROGRESS
     STATUS _status)
list(GET _status 0 _code)
list(GET _status 1 _message)
if(NOT _code EQUAL 0)
  file(REMOVE "${_archive}")
  message(FATAL_ERROR "${NAME}: download of ${URL} failed: ${_message}")
endif()

file(SHA256 "${_archive}" _actual)
if(NOT _actual STREQUAL SHA256)
  file(REMOVE "${_archive}")
  message(FATAL_ERROR
    "${NAME}: ${URL} has SHA256 ${_actual}, but Testing/datasets/${NAME}.cmake "
    "expects ${SHA256}. The file on the server changed or the download is "
    "corrupt; nothing was unpacked.")
endif()

# Unpack next to DEST first, so DEST only ever holds a complete set.
file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${_unpack}")
file(REMOVE "${_archive}")
file(REMOVE_RECURSE "${DEST}")
file(RENAME "${_unpack}" "${DEST}")
file(WRITE "${_stamp}" "${SHA256}\n")
message(STATUS "${NAME}: unpacked to ${DEST}")
