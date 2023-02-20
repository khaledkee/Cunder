cmake_minimum_required(VERSION 3.16)

set(TEST_TARGET_NAME test-cunder)

add_executable(${TEST_TARGET_NAME} acutest.h test-cunder.c)
target_link_libraries(${TEST_TARGET_NAME} PRIVATE cunder)
target_include_directories(${TEST_TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/Cunder)