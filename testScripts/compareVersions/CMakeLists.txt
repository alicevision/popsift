set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

find_program(AWK awk mawk gawk)
find_program(SORT sort)
find_program(GNUPLOT gnuplot)
find_program(CONVERT convert)
find_program(VLFEAT sift PATH_SUFFIXES glnxa64 bin/glnxa64)

if(GNUPLOT MATCHES ".+-NOTFOUND")
    message(WARNING  "WARNING: gnupot could not be found (${GNUPLOT}).")
else()
    message(STATUS  "gnuplot found (${GNUPLOT}).")
endif()

if(CONVERT MATCHES ".+-NOTFOUND")
    message(WARNING  "WARNING: ImageMagick's convert could not be found (${CONVERT}).")
else()
    message(STATUS  "convert found (${CONVERT}).")
endif()

if(VLFEAT MATCHES ".+-NOTFOUND")
    message(WARNING  "WARNING: vlfeat could not be found (${VLFEAT}).")
else()
    message(STATUS  "vlfeat found (${VLFEAT}).")
endif()

if(SORT MATCHES ".+-NOTFOUND")
    message(WARNING  "WARNING: sort could not be found (${SORT}). Compare tests will fail.")
else()
    message(STATUS  "sort found (${SORT}).")
endif()

if(AWK MATCHES ".+-NOTFOUND")
    message(WARNING  "WARNING: awk (and mawk and gawk) could not be found (${AWK}). Compare tests will fail")
else()
    message(STATUS  "awk found (${AWK}).")
endif()

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/compareWithVLFeat.sh.in
                ${CMAKE_CURRENT_BINARY_DIR}/compareWithVLFeat.sh
		@ONLY)

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/Compare.mk.in
                Compare.mk
		@ONLY)

