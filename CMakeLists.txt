cmake_minimum_required(VERSION 3.12)
project(15418_project)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)

# Specify the source files (main.cpp)
set(SOURCE_FILES main.cpp)

# Specify the include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Specify the library directories
link_directories(${CMAKE_SOURCE_DIR}/lib)

# Add the executable
add_executable(renderer ${SOURCE_FILES})

# set(CMAKE_PREFIX_PATH "/home/johnsmith/Downloads/glfw-3.3.8/cmake_dependencies")
set(glfw3_DIR "/home/johnsmith/Downloads/glfw-3.3.8/cmake_dependencies/")
# set(glfw3_DIR "/home/johnsmith/Downloads/glfw-3.3.8/build/src")

find_package(glfw3 REQUIRED)
# Link against the GLFW library
# target_link_libraries(renderer glfw)
# find_package(OpenGL REQUIRED)
# target_link_libraries(renderer OpenGL::GL)
target_link_libraries(renderer libglfw3.a)