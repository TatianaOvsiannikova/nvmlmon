cmake_minimum_required(VERSION 3.10)
project(GpuMonitor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")
# -------------------------------
# nlohmann_json
# -------------------------------
find_package(nlohmann_json REQUIRED)

# -------------------------------
# Discover CUDA installations
# -------------------------------
file(GLOB CUDA_ROOT_DIRS "/usr/local/cuda*" "/usr/local/cuda")
list(REMOVE_DUPLICATES CUDA_ROOT_DIRS)

message(STATUS "CUDA root candidates: ${CUDA_ROOT_DIRS}")

# -------------------------------
# Find NVML header (nvml.h)
# -------------------------------
find_path(NVML_INCLUDE_DIR
    NAMES nvml.h
    PATHS ${CUDA_ROOT_DIRS} /usr /usr/local
    PATH_SUFFIXES
        include
        targets/x86_64-linux/include
)

if(NOT NVML_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find nvml.h in any CUDA installation")
endif()

message(STATUS "Found NVML include dir: ${NVML_INCLUDE_DIR}")

# -------------------------------
# Find NVML library
# Prefer system (real) NVML library over CUDA stubs
# -------------------------------
find_library(NVML_LIB
    NAMES nvidia-ml libnvidia-ml
    HINTS
        /usr/lib64
        /usr/lib
        /usr/lib/x86_64-linux-gnu
)

# Fallback: search CUDA stub NVML if driver NVML not found
if(NOT NVML_LIB)
    message(WARNING "Real NVML library not found — falling back to CUDA stubs")

    find_library(NVML_LIB
        NAMES nvidia-ml
        PATHS ${CUDA_ROOT_DIRS}
        PATH_SUFFIXES targets/x86_64-linux/lib/stubs lib64/stubs
    )
endif()

if(NOT NVML_LIB)
    message(FATAL_ERROR "NVML library was not found (neither real nor stub)")
endif()

message(STATUS "Using NVML library: ${NVML_LIB}")

# -------------------------------
# Executable
# -------------------------------
set(SRC_FILES
    src/main.cpp
    src/nvmlmon.cpp
)

add_executable(nvmlmon ${SRC_FILES})

target_include_directories(nvmlmon PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${NLOHMANN_JSON_INCLUDE_DIRS}
    ${NVML_INCLUDE_DIR}
)

target_link_libraries(nvmlmon PRIVATE
    nlohmann_json::nlohmann_json
    ${NVML_LIB}
)

# -------------------------------
# Install
# -------------------------------
install(TARGETS nvmlmon RUNTIME DESTINATION bin)

