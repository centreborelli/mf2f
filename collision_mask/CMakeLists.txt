cmake_policy(SET CMP0048 NEW) # avoid annoing warnings

project(occlu-mask VERSION 1.0 LANGUAGES C)
message(STATUS "Configuring main project: " ${CMAKE_PROJECT_NAME})

# flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -ggdb -fsanitize=address")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall -ggdb -O6 -w -s -ffast-math -fno-finite-math-only -fomit-frame-pointer -fstrength-reduce")
set(CMAKE_BUILD_TYPE Release)

# cmake global
cmake_minimum_required(VERSION 2.8.2)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# dependencies
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# add project directories
add_subdirectory(iio)
add_subdirectory(argparse)

# enable C99
if (CMAKE_VERSION VERSION_LESS "3.1")
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
else ()
	set (CMAKE_C_STANDARD 99)
endif ()

# internal includes
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${ARGPARSE_INCLUDE_DIR})
include_directories(${IIO_INCLUDE_DIR})

# set targets
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable(occlu-mask occlu-mask.c)
add_executable(warp-bicubic warp-bicubic.c)
link_directories(${CMAKE_BINARY_DIR}/lib)
target_link_libraries(occlu-mask iio argparse)
target_link_libraries(warp-bicubic iio argparse)
