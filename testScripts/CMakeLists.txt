configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/downloadOxfordDataset.sh.in
                ${CMAKE_CURRENT_BINARY_DIR}/downloadOxfordDataset.sh )

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/testOxfordDataset.sh.in
                ${CMAKE_CURRENT_BINARY_DIR}/testOxfordDataset.sh )

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/TEST.sh.in
                ${CMAKE_CURRENT_BINARY_DIR}/TEST.sh )

add_custom_target(
	prepare-test
 	COMMAND /bin/bash ${CMAKE_CURRENT_BINARY_DIR}/downloadOxfordDataset.sh
 	DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/downloadOxfordDataset.sh
)

add_custom_target(
	run-test
 	COMMAND /bin/bash ${CMAKE_CURRENT_BINARY_DIR}/testOxfordDataset.sh
 	DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/testOxfordDataset.sh
	DEPENDS popsift-demo
)

add_custom_target(
	run-test-boat
 	COMMAND /bin/bash ${CMAKE_CURRENT_BINARY_DIR}/testOxfordDataset.sh boat
 	DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/testOxfordDataset.sh
	DEPENDS popsift-demo
)

add_subdirectory(compareVersions)

