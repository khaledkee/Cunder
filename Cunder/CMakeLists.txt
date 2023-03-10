cmake_minimum_required(VERSION 3.16)

add_library(cunder SHARED c_libtorch.h c_libtorch.cpp)

# for dllexport on WIN32
target_compile_definitions(cunder PRIVATE "CUNDER_COMPILE_LIBRARY")

target_link_libraries(cunder PRIVATE "${TORCH_LIBRARIES}")

if(WIN32)
    # The following code block is suggested to be used on Windows.
    # According to https://github.com/pytorch/pytorch/issues/25457,
    # the DLLs need to be copied to avoid memory errors.
    file(GLOB TORCH_DLLS "${LIBTORCH_PATH}/lib/*.dll")
    add_custom_command(TARGET cunder
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)
endif()