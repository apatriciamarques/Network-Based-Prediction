cmake_minimum_required(VERSION 3.18)  # CUDA requires >= 3.18
project(Pipeline LANGUAGES CXX CUDA)  # Add CUDA language support

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)  # Set CUDA standard
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable CUDA separable compilation
set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Find required packages
find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ZLIB REQUIRED)
find_package(CUDAToolkit REQUIRED)  # Use modern CUDA toolkit finder

message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Include Dirs: ${CUDAToolkit_INCLUDE_DIRS}")

# Include directories
include_directories(
    /usr/local/include   # for cnpy.h
    src                  # your project headers
)

# Collect sources - separate C++ and CUDA files
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
file(GLOB_RECURSE HEADERS "src/*.h")

# Debug output
message(STATUS "Found C++ sources: ${CPP_SOURCES}")
message(STATUS "Found CUDA sources: ${CUDA_SOURCES}")

# Create executable with both C++ and CUDA sources
add_executable(${PROJECT_NAME} 
    ${CPP_SOURCES} 
    ${CUDA_SOURCES}
    ${HEADERS}
)

# Set CUDA properties for the target
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "60;70;75;80;86"  # Support multiple GPU architectures
)

# Link libraries using modern CMake targets
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${BLAS_LIBRARIES}
    /usr/local/lib/libcnpy.a
    ZLIB::ZLIB
    OpenMP::OpenMP_CXX
    CUDA::cudart         # CUDA runtime
    CUDA::cublas         # cuBLAS library
    CUDA::curand         # cuRAND library (if needed)
)

# Compiler options for C++
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
)

# Compiler options for CUDA
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# Optional: Add debug information for CUDA in debug mode
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>
)

# Print some useful information
message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
