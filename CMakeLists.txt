# cmake_minimum_required(VERSION 3.12)
# project(15418_project)

# set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}")

# set(CMAKE_CXX_STANDARD 14)
# file(GLOB_RECURSE SOURCES "src/*.cpp")

# find_package(OpenGL REQUIRED)
# find_package(GLEW REQUIRED)

# add_executable(renderer ${SOURCES})

# target_include_directories(renderer PRIVATE "${CMAKE_SOURCE_DIR}/src/include")
# target_link_libraries(renderer PRIVATE glfw GL)

cmake_minimum_required(VERSION 3.12)
project(YourProjectName)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories("${CMAKE_SOURCE_DIR}/src/include")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}")

add_executable(renderer src/main.cpp src/glad.c)

target_link_libraries(renderer ${OPENGL_LIBRARIES} GLEW::GLEW)

target_link_libraries(renderer glfw)