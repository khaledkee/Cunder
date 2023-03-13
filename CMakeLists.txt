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
find_package(Torch REQUIRED PATHS "${LIBTORCH_PATH}" NO_DEFAULT_PATH CONFIG)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include(FetchContent)

# https://github.com/Maverobot/libtorch_examples/blob/master/CMakeLists.txt
find_package(Torch CONFIG)
if(NOT Torch_FOUND)
  message(STATUS "libtorch not found")
  message(STATUS "fetching libtorch cpu-1.13.1")
FetchContent_Declare(
    libtorch
    URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip
    SOURCE_DIR libtorch)
  FetchContent_GetProperties(libtorch)
  if(NOT libtorch_POPULATED)
    unset(FETCHCONTENT_QUIET CACHE)
    FetchContent_Populate(libtorch)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/libtorch)
  endif()
  find_package(Torch REQUIRED PATHS "${CMAKE_BINARY_DIR}/libtorch" NO_DEFAULT_PATH CONFIG)
endif()

set(CUNDER_DATA_DIR "${CMAKE_SOURCE_DIR}/cunder-data")
add_subdirectory(Cunder)

if(CUNDER_BUILD_TESTS)
    FetchContent_Declare(
        doctest
        GIT_REPOSITORY https://github.com/onqtam/doctest.git
        GIT_TAG        2.4.6
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(doctest)

    add_subdirectory(playground)
    add_subdirectory(unittests)
endif()
