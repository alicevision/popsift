find_package(Doxygen REQUIRED)

file(GLOB_RECURSE ALL_PUBLIC_HEADERS ${PROJECT_SOURCE_DIR}/src/popsift/*.h*)
set(CCTAG_OTHER_DOC_SOURCES ${PROJECT_SOURCE_DIR}/README.md ${PROJECT_SOURCE_DIR}/INSTALL.md)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${PROJECT_SOURCE_DIR}/README.md)
set(DOXYGEN_PROJECT_BRIEF "A faithful implementation of the SIFT algorithm in CUDA.")
set(DOXYGEN_GENERATE_XML YES)
set(DOXYGEN_GENERATE_TREEVIEW YES)
set(DOXYGEN_GENERATE_DEPRECATEDLIST YES)
set(DOXYGEN_SORT_BRIEF_DOCS YES)
set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIRECTORY}/xml/index.xml)

doxygen_add_docs(doxygen
        ${ALL_PUBLIC_HEADERS} ${CCTAG_OTHER_DOC_SOURCES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generate the doc")



find_package(Sphinx REQUIRED)

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/source)
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)

add_custom_target(sphinx ALL
        COMMAND
        ${SPHINX_EXECUTABLE} -b html
        # Tell Breathe where to find the Doxygen output
        -Dbreathe_projects.PopSift=${DOXYGEN_OUTPUT_DIRECTORY}/xml
        ${SPHINX_SOURCE} ${SPHINX_BUILD}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS
        doxygen
        # Other docs files you want to track should go here (or in some variable)
        ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/source/index.rst
        ${DOXYGEN_INDEX_FILE}
        #        MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
        COMMENT "Generating documentation with Sphinx")