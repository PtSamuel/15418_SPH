cmake_minimum_required(VERSION 3.12)
project(15418_project)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 14)
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(renderer ${SOURCES})
target_link_libraries(renderer PRIVATE glfw)