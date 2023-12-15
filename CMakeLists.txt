cmake_minimum_required(VERSION 3.12)
project(15418_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

add_executable(renderer_debug src/main.cpp)
add_executable(renderer_gpu_bitonic src/main_gpu.cpp src/gpu.cu src/bitonic_sort.cu)
add_executable(test_bitonic_sort src/bitonic_sort.cu src/test_bitonic_sort.cpp)
add_executable(renderer_cpu_blocks src/main_cpu_blocks.cpp)
add_executable(renderer_gpu_blocks src/main_gpu_blocks.cpp src/gpu_blocks.cu)

include_directories("${CMAKE_SOURCE_DIR}/src/include")
include_directories(${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}")

target_link_libraries(renderer_debug ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})
target_link_libraries(renderer_gpu_bitonic ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(renderer_cpu_blocks ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(renderer_gpu_blocks ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${CUDA_LIBRARIES})

target_link_libraries(renderer_debug glfw)
target_link_libraries(renderer_gpu_bitonic glfw)
target_link_libraries(renderer_cpu_blocks glfw)
target_link_libraries(renderer_gpu_blocks glfw)
