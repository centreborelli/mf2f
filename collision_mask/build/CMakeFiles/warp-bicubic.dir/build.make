# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/dewil/occlu-mask

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dewil/occlu-mask/build

# Include any dependencies generated for this target.
include CMakeFiles/warp-bicubic.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/warp-bicubic.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/warp-bicubic.dir/flags.make

CMakeFiles/warp-bicubic.dir/warp-bicubic.c.o: CMakeFiles/warp-bicubic.dir/flags.make
CMakeFiles/warp-bicubic.dir/warp-bicubic.c.o: ../warp-bicubic.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dewil/occlu-mask/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/warp-bicubic.dir/warp-bicubic.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/warp-bicubic.dir/warp-bicubic.c.o   -c /home/dewil/occlu-mask/warp-bicubic.c

CMakeFiles/warp-bicubic.dir/warp-bicubic.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/warp-bicubic.dir/warp-bicubic.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/dewil/occlu-mask/warp-bicubic.c > CMakeFiles/warp-bicubic.dir/warp-bicubic.c.i

CMakeFiles/warp-bicubic.dir/warp-bicubic.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/warp-bicubic.dir/warp-bicubic.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/dewil/occlu-mask/warp-bicubic.c -o CMakeFiles/warp-bicubic.dir/warp-bicubic.c.s

CMakeFiles/warp-bicubic.dir/warp-bicubic.c.o.requires:

.PHONY : CMakeFiles/warp-bicubic.dir/warp-bicubic.c.o.requires

CMakeFiles/warp-bicubic.dir/warp-bicubic.c.o.provides: CMakeFiles/warp-bicubic.dir/warp-bicubic.c.o.requires
	$(MAKE) -f CMakeFiles/warp-bicubic.dir/build.make CMakeFiles/warp-bicubic.dir/warp-bicubic.c.o.provides.build
.PHONY : CMakeFiles/warp-bicubic.dir/warp-bicubic.c.o.provides

CMakeFiles/warp-bicubic.dir/warp-bicubic.c.o.provides.build: CMakeFiles/warp-bicubic.dir/warp-bicubic.c.o


# Object files for target warp-bicubic
warp__bicubic_OBJECTS = \
"CMakeFiles/warp-bicubic.dir/warp-bicubic.c.o"

# External object files for target warp-bicubic
warp__bicubic_EXTERNAL_OBJECTS =

bin/warp-bicubic: CMakeFiles/warp-bicubic.dir/warp-bicubic.c.o
bin/warp-bicubic: CMakeFiles/warp-bicubic.dir/build.make
bin/warp-bicubic: iio/libiio.a
bin/warp-bicubic: argparse/libargparse.a
bin/warp-bicubic: CMakeFiles/warp-bicubic.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dewil/occlu-mask/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable bin/warp-bicubic"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/warp-bicubic.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/warp-bicubic.dir/build: bin/warp-bicubic

.PHONY : CMakeFiles/warp-bicubic.dir/build

CMakeFiles/warp-bicubic.dir/requires: CMakeFiles/warp-bicubic.dir/warp-bicubic.c.o.requires

.PHONY : CMakeFiles/warp-bicubic.dir/requires

CMakeFiles/warp-bicubic.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/warp-bicubic.dir/cmake_clean.cmake
.PHONY : CMakeFiles/warp-bicubic.dir/clean

CMakeFiles/warp-bicubic.dir/depend:
	cd /home/dewil/occlu-mask/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dewil/occlu-mask /home/dewil/occlu-mask /home/dewil/occlu-mask/build /home/dewil/occlu-mask/build /home/dewil/occlu-mask/build/CMakeFiles/warp-bicubic.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/warp-bicubic.dir/depend

