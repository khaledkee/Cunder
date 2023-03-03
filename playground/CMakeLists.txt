cmake_minimum_required(VERSION 3.16)

set(PLAYGROUND_TARGET_NAME playground)

add_executable(${PLAYGROUND_TARGET_NAME} main.cpp)
target_link_libraries(${PLAYGROUND_TARGET_NAME} PRIVATE cunder)
target_include_directories(${PLAYGROUND_TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/Cunder)


target_compile_definitions(${PLAYGROUND_TARGET_NAME}
    PUBLIC
        -DCUNDER_DATA_DIR="${CUNDER_DATA_DIR}"
)
