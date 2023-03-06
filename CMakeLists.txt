cmake_minimum_required(VERSION 3.16)

set(MASTER_PROJECT OFF)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(MASTER_PROJECT ON)
endif()

project(cunder LANGUAGES CXX C)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
option(CUNDER_BUILD_TESTS "Build unit tests" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# TODO: don't use absolute path
IF(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(LIBTORCH_PATH "D:/Libtorch/Original/libtorch-win-shared-with-deps-debug-1.13.1+cpu/libtorch")
ELSE()
    set(LIBTORCH_PATH "D:/Libtorch/Original/libtorch-win-shared-with-deps-1.13.1+cpu/libtorch")
ENDIF()
find_package(Torch REQUIRED PATHS "${LIBTORCH_PATH}" NO_DEFAULT_PATH)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(Cunder)

if(CUNDER_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/onqtam/doctest.git
    GIT_TAG        2.4.6 # 2.4.6
    GIT_SHALLOW    TRUE
    EXCLUDE_FROM_ALL TRUE
    )
    FetchContent_MakeAvailable(doctest)


    set(CUNDER_DATA_DIR "${CMAKE_SOURCE_DIR}/cunder-data")
    add_subdirectory(playground)
    add_subdirectory(unittests)
endif()
