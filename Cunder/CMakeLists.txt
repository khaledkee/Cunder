cmake_minimum_required(VERSION 3.16)

add_library(cunder SHARED
	c_libtorch.h
    c_libtorch.cpp
)

# for dllexport on WIN32
target_compile_definitions(cunder PRIVATE "CUNDER_COMPILE_LIBRARY")

target_link_libraries(cunder
)

target_compile_options(cunder
    PRIVATE -Wall -pedantic -fno-omit-frame-pointer -O3
)