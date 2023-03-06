cmake_minimum_required(VERSION 3.16)

set(TEST_TARGET_NAME test-cunder)

add_executable(${TEST_TARGET_NAME} test-main.cpp test-cunder.cpp)
target_link_libraries(${TEST_TARGET_NAME} PRIVATE cunder doctest::doctest)
target_include_directories(${TEST_TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/Cunder)

target_compile_definitions(${TEST_TARGET_NAME}
    PUBLIC -DCUNDER_DATA_DIR="${CUNDER_DATA_DIR}"
)