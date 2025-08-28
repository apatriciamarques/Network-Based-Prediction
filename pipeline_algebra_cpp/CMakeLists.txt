cmake_minimum_required(VERSION 3.10)
project(Pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find BLAS library
find_package(BLAS REQUIRED)

# Find OpenMP (optional)

find_package(OpenMP REQUIRED)

# Find ZLIB
find_package(ZLIB REQUIRED)

# Include directories
include_directories(
    /usr/local/include   # for cnpy.h
    src                  # your project headers
)

# Collect sources
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "src/*.h")

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${BLAS_LIBRARIES}
    /usr/local/lib/libcnpy.a  # link system-installed cnpy
    ZLIB::ZLIB                # link zlib
    OpenMP::OpenMP_CXX
)

# Compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE
    -O3
    -march=native
    -ffast-math
)

target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
