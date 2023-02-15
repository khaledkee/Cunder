cmake_minimum_required(VERSION 3.16)

set(MASTER_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(MASTER_PROJECT ON)
endif ()

project(cunder LANGUAGES CXX C)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
option(CUNDER_BUILD_TESTS "Build unit tests" OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


if (CUNDER_BUILD_TESTS)
#    include(CPM)
    # doctest
#    CPMDeclarePackage(doctest
#      NAME doctest
#      GIT_TAG 2.4.6
#      GIT_REPOSITORY https://github.com/onqtam/doctest.git
#      GIT_SHALLOW TRUE
#      EXCLUDE_FROM_ALL TRUE
#    )
#    add_subdirectory(unittests)

    if (MASTER_PROJECT)
        add_subdirectory(playground)
    endif()
endif()
