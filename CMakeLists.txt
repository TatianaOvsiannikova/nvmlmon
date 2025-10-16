cmake_minimum_required(VERSION 3.10)
project(GpuMonitor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(nlohmann_json REQUIRED)

# Source files
set(SRC_FILES
    src/main.cpp
    src/nvmlmon.cpp
)

# Define the executable
add_executable(nvmlmon ${SRC_FILES})

# Include directories
target_include_directories(nvmlmon PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${NLOHMANN_JSON_INCLUDE_DIRS}
)
find_library(NVML_LIB nvidia-ml
  HINTS
    /usr/lib64
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
    /usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs
    /usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs
)
# Link libraries
target_link_libraries(nvmlmon PRIVATE
    nlohmann_json::nlohmann_json
     ${NVML_LIB}
)

# Optional: install target
install(TARGETS nvmlmon RUNTIME DESTINATION bin)


