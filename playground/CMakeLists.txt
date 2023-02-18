cmake_minimum_required(VERSION 3.16)

set(TEST_TARGET_NAME playground)
set(CMAKE_PREFIX_PATH "D:\\Libtorch\\Original\\libtorch-win-shared-with-deps-1.13.1+cpu\\libtorch")
find_package(Torch REQUIRED)

add_executable(${TEST_TARGET_NAME} main.cpp)
target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(${TEST_TARGET_NAME} PRIVATE cunder)
target_include_directories(${TEST_TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/Cunder)