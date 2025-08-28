cmake_minimum_required(VERSION 3.10)
# Project name and language
project(Pipeline LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find BLAS library
find_package(BLAS REQUIRED)

# Find OpenMP for parallelization (optional but recommended)
find_package(OpenMP)

# Collect all source and header files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "src/*.h")

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link BLAS library
target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARIES})

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

# Add compiler flags for optimization
target_compile_options(${PROJECT_NAME} PRIVATE
    -O3
    -march=native
    -ffast-math
)

# Alternative: If you prefer to link BLAS manually
# target_link_libraries(${PROJECT_NAME} blas)
# or
# target_link_libraries(${PROJECT_NAME} openblas)
# or for Intel MKL:
# target_link_libraries(${PROJECT_NAME} mkl_intel_lp64 mkl_gnu_thread mkl_core)
