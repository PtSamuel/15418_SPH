# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/johnsmith/Desktop/15418_project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/johnsmith/Desktop/15418_project/build

# Include any dependencies generated for this target.
include CMakeFiles/renderer_gpu.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/renderer_gpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/renderer_gpu.dir/flags.make

CMakeFiles/renderer_gpu.dir/src/main_gpu.cpp.o: CMakeFiles/renderer_gpu.dir/flags.make
CMakeFiles/renderer_gpu.dir/src/main_gpu.cpp.o: ../src/main_gpu.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/johnsmith/Desktop/15418_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/renderer_gpu.dir/src/main_gpu.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/renderer_gpu.dir/src/main_gpu.cpp.o -c /home/johnsmith/Desktop/15418_project/src/main_gpu.cpp

CMakeFiles/renderer_gpu.dir/src/main_gpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/renderer_gpu.dir/src/main_gpu.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/johnsmith/Desktop/15418_project/src/main_gpu.cpp > CMakeFiles/renderer_gpu.dir/src/main_gpu.cpp.i

CMakeFiles/renderer_gpu.dir/src/main_gpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/renderer_gpu.dir/src/main_gpu.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/johnsmith/Desktop/15418_project/src/main_gpu.cpp -o CMakeFiles/renderer_gpu.dir/src/main_gpu.cpp.s

CMakeFiles/renderer_gpu.dir/src/gpu.cu.o: CMakeFiles/renderer_gpu.dir/flags.make
CMakeFiles/renderer_gpu.dir/src/gpu.cu.o: ../src/gpu.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/johnsmith/Desktop/15418_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/renderer_gpu.dir/src/gpu.cu.o"
	/usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/johnsmith/Desktop/15418_project/src/gpu.cu -o CMakeFiles/renderer_gpu.dir/src/gpu.cu.o

CMakeFiles/renderer_gpu.dir/src/gpu.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/renderer_gpu.dir/src/gpu.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/renderer_gpu.dir/src/gpu.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/renderer_gpu.dir/src/gpu.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target renderer_gpu
renderer_gpu_OBJECTS = \
"CMakeFiles/renderer_gpu.dir/src/main_gpu.cpp.o" \
"CMakeFiles/renderer_gpu.dir/src/gpu.cu.o"

# External object files for target renderer_gpu
renderer_gpu_EXTERNAL_OBJECTS =

../renderer_gpu: CMakeFiles/renderer_gpu.dir/src/main_gpu.cpp.o
../renderer_gpu: CMakeFiles/renderer_gpu.dir/src/gpu.cu.o
../renderer_gpu: CMakeFiles/renderer_gpu.dir/build.make
../renderer_gpu: /usr/lib/x86_64-linux-gnu/libOpenGL.so
../renderer_gpu: /usr/lib/x86_64-linux-gnu/libGLX.so
../renderer_gpu: /usr/lib/x86_64-linux-gnu/libGLU.so
../renderer_gpu: /usr/lib/x86_64-linux-gnu/libGLEW.so
../renderer_gpu: /usr/lib/x86_64-linux-gnu/libcudart_static.a
../renderer_gpu: /usr/lib/x86_64-linux-gnu/librt.so
../renderer_gpu: CMakeFiles/renderer_gpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/johnsmith/Desktop/15418_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../renderer_gpu"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/renderer_gpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/renderer_gpu.dir/build: ../renderer_gpu

.PHONY : CMakeFiles/renderer_gpu.dir/build

CMakeFiles/renderer_gpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/renderer_gpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/renderer_gpu.dir/clean

CMakeFiles/renderer_gpu.dir/depend:
	cd /home/johnsmith/Desktop/15418_project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/johnsmith/Desktop/15418_project /home/johnsmith/Desktop/15418_project /home/johnsmith/Desktop/15418_project/build /home/johnsmith/Desktop/15418_project/build /home/johnsmith/Desktop/15418_project/build/CMakeFiles/renderer_gpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/renderer_gpu.dir/depend

