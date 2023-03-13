cmake_minimum_required(VERSION 3.16)

project(test-cunder)

add_executable(${PROJECT_NAME} test-main.cpp test-cunder.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE cunder doctest::doctest)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/Cunder)

target_compile_definitions(${PROJECT_NAME}
    PUBLIC -DCUNDER_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/data"
)