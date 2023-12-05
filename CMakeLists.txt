cmake_minimum_required(VERSION 3.12)
project(15418_project)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

add_executable(renderer src/main.cpp)

include_directories("${CMAKE_SOURCE_DIR}/src/include")
include_directories(${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})
target_link_libraries(renderer ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}")
target_link_libraries(renderer glfw)