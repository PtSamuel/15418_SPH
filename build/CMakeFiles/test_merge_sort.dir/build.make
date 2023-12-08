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
include CMakeFiles/test_merge_sort.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_merge_sort.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_merge_sort.dir/flags.make

CMakeFiles/test_merge_sort.dir/src/bitonic_sort.cu.o: CMakeFiles/test_merge_sort.dir/flags.make
CMakeFiles/test_merge_sort.dir/src/bitonic_sort.cu.o: ../src/bitonic_sort.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/johnsmith/Desktop/15418_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/test_merge_sort.dir/src/bitonic_sort.cu.o"
	/usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/johnsmith/Desktop/15418_project/src/bitonic_sort.cu -o CMakeFiles/test_merge_sort.dir/src/bitonic_sort.cu.o

CMakeFiles/test_merge_sort.dir/src/bitonic_sort.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/test_merge_sort.dir/src/bitonic_sort.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/test_merge_sort.dir/src/bitonic_sort.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/test_merge_sort.dir/src/bitonic_sort.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/test_merge_sort.dir/src/test_bitonic_sort.cpp.o: CMakeFiles/test_merge_sort.dir/flags.make
CMakeFiles/test_merge_sort.dir/src/test_bitonic_sort.cpp.o: ../src/test_bitonic_sort.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/johnsmith/Desktop/15418_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/test_merge_sort.dir/src/test_bitonic_sort.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_merge_sort.dir/src/test_bitonic_sort.cpp.o -c /home/johnsmith/Desktop/15418_project/src/test_bitonic_sort.cpp

CMakeFiles/test_merge_sort.dir/src/test_bitonic_sort.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_merge_sort.dir/src/test_bitonic_sort.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/johnsmith/Desktop/15418_project/src/test_bitonic_sort.cpp > CMakeFiles/test_merge_sort.dir/src/test_bitonic_sort.cpp.i

CMakeFiles/test_merge_sort.dir/src/test_bitonic_sort.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_merge_sort.dir/src/test_bitonic_sort.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/johnsmith/Desktop/15418_project/src/test_bitonic_sort.cpp -o CMakeFiles/test_merge_sort.dir/src/test_bitonic_sort.cpp.s

# Object files for target test_merge_sort
test_merge_sort_OBJECTS = \
"CMakeFiles/test_merge_sort.dir/src/bitonic_sort.cu.o" \
"CMakeFiles/test_merge_sort.dir/src/test_bitonic_sort.cpp.o"

# External object files for target test_merge_sort
test_merge_sort_EXTERNAL_OBJECTS =

../test_merge_sort: CMakeFiles/test_merge_sort.dir/src/bitonic_sort.cu.o
../test_merge_sort: CMakeFiles/test_merge_sort.dir/src/test_bitonic_sort.cpp.o
../test_merge_sort: CMakeFiles/test_merge_sort.dir/build.make
../test_merge_sort: CMakeFiles/test_merge_sort.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/johnsmith/Desktop/15418_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../test_merge_sort"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_merge_sort.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_merge_sort.dir/build: ../test_merge_sort

.PHONY : CMakeFiles/test_merge_sort.dir/build

CMakeFiles/test_merge_sort.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_merge_sort.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_merge_sort.dir/clean

CMakeFiles/test_merge_sort.dir/depend:
	cd /home/johnsmith/Desktop/15418_project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/johnsmith/Desktop/15418_project /home/johnsmith/Desktop/15418_project /home/johnsmith/Desktop/15418_project/build /home/johnsmith/Desktop/15418_project/build /home/johnsmith/Desktop/15418_project/build/CMakeFiles/test_merge_sort.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_merge_sort.dir/depend

