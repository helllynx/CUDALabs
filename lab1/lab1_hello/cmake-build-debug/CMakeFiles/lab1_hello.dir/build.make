# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

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
CMAKE_COMMAND = /home/black/SOFT/clion-2018.2.3/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/black/SOFT/clion-2018.2.3/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/black/MIET/CUDA/lab1/lab1_hello

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/black/MIET/CUDA/lab1/lab1_hello/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/lab1_hello.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lab1_hello.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lab1_hello.dir/flags.make

CMakeFiles/lab1_hello.dir/main.cu.o: CMakeFiles/lab1_hello.dir/flags.make
CMakeFiles/lab1_hello.dir/main.cu.o: ../main.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/black/MIET/CUDA/lab1/lab1_hello/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/lab1_hello.dir/main.cu.o"
	/usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/black/MIET/CUDA/lab1/lab1_hello/main.cu -o CMakeFiles/lab1_hello.dir/main.cu.o

CMakeFiles/lab1_hello.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/lab1_hello.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/lab1_hello.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/lab1_hello.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target lab1_hello
lab1_hello_OBJECTS = \
"CMakeFiles/lab1_hello.dir/main.cu.o"

# External object files for target lab1_hello
lab1_hello_EXTERNAL_OBJECTS =

CMakeFiles/lab1_hello.dir/cmake_device_link.o: CMakeFiles/lab1_hello.dir/main.cu.o
CMakeFiles/lab1_hello.dir/cmake_device_link.o: CMakeFiles/lab1_hello.dir/build.make
CMakeFiles/lab1_hello.dir/cmake_device_link.o: CMakeFiles/lab1_hello.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/black/MIET/CUDA/lab1/lab1_hello/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/lab1_hello.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lab1_hello.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lab1_hello.dir/build: CMakeFiles/lab1_hello.dir/cmake_device_link.o

.PHONY : CMakeFiles/lab1_hello.dir/build

# Object files for target lab1_hello
lab1_hello_OBJECTS = \
"CMakeFiles/lab1_hello.dir/main.cu.o"

# External object files for target lab1_hello
lab1_hello_EXTERNAL_OBJECTS =

lab1_hello: CMakeFiles/lab1_hello.dir/main.cu.o
lab1_hello: CMakeFiles/lab1_hello.dir/build.make
lab1_hello: CMakeFiles/lab1_hello.dir/cmake_device_link.o
lab1_hello: CMakeFiles/lab1_hello.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/black/MIET/CUDA/lab1/lab1_hello/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable lab1_hello"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lab1_hello.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lab1_hello.dir/build: lab1_hello

.PHONY : CMakeFiles/lab1_hello.dir/build

CMakeFiles/lab1_hello.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lab1_hello.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lab1_hello.dir/clean

CMakeFiles/lab1_hello.dir/depend:
	cd /home/black/MIET/CUDA/lab1/lab1_hello/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/black/MIET/CUDA/lab1/lab1_hello /home/black/MIET/CUDA/lab1/lab1_hello /home/black/MIET/CUDA/lab1/lab1_hello/cmake-build-debug /home/black/MIET/CUDA/lab1/lab1_hello/cmake-build-debug /home/black/MIET/CUDA/lab1/lab1_hello/cmake-build-debug/CMakeFiles/lab1_hello.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lab1_hello.dir/depend

