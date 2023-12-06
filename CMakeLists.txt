cmake_minimum_required(VERSION 3.12)
project(15418_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)


add_executable(renderer src/main.cpp)
add_executable(renderer_gpu src/main_gpu.cpp src/gpu.cu)

include_directories("${CMAKE_SOURCE_DIR}/src/include")
include_directories(${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})
target_link_libraries(renderer ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})
target_link_libraries(renderer_gpu ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${CUDA_LIBRARIES})

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}")
target_link_libraries(renderer glfw)
target_link_libraries(renderer_gpu glfw)