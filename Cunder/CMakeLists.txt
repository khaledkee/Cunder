cmake_minimum_required(VERSION 3.16)

add_library(cunder SHARED
		include/Cunder/c_libtorch.h
    src/Cunder/c_libtorch.cpp
)

set_target_properties(cunder PROPERTIES PREFIX "")
# for dllexport on WIN32
target_compile_definitions(cunder PRIVATE "CUNDER_COMPILE_LIBRARY")

target_link_libraries(cunder
)

target_compile_options(cunder
    PRIVATE -Wall -Wextra -pedantic -fno-omit-frame-pointer -O3 -Wno-unused-variable -Wno-documentation -Wno-padded -Wno-c++98-compat-pedantic
)