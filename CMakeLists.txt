cmake_minimum_required(VERSION 3.10)
project(MiniTensor VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Eigen3
# Direct path for macOS Homebrew installation
set(EIGEN3_INCLUDE_DIR "/opt/homebrew/Cellar/eigen/5.0.1/include/eigen3")

# Verify the path exists
if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Dense")
    message(WARNING "Eigen3 not found at hardcoded path: ${EIGEN3_INCLUDE_DIR}")
    message(STATUS "Trying to find Eigen3 in common locations...")
    
    # Fallback: try common installation paths
    set(EIGEN3_POSSIBLE_PATHS
        /opt/homebrew/include/eigen3
        /usr/local/include/eigen3
        /usr/include/eigen3
        /usr/local/include
        /usr/include
        /opt/homebrew/include
    )
    
    set(EIGEN3_FOUND FALSE)
    foreach(PATH ${EIGEN3_POSSIBLE_PATHS})
        if(EXISTS "${PATH}/Eigen/Dense")
            set(EIGEN3_INCLUDE_DIR "${PATH}")
            set(EIGEN3_FOUND TRUE)
            message(STATUS "Found Eigen3 at: ${EIGEN3_INCLUDE_DIR}")
            break()
        endif()
    endforeach()
    
    if(NOT EIGEN3_FOUND)
        message(FATAL_ERROR "Eigen3 not found. Please install Eigen3 using: brew install eigen")
    endif()
else()
    message(STATUS "Using Eigen3 at: ${EIGEN3_INCLUDE_DIR}")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Source files
set(MINITENSOR_SOURCES
    src/core.cpp
    src/functions.cpp
    src/layers.cpp
    src/models.cpp
    src/optimizers.cpp
)

# Header files (for IDE support)
set(MINITENSOR_HEADERS
    include/miniTensor/core.h
    include/miniTensor/functions.h
    include/miniTensor/layers.h
    include/miniTensor/models.h
    include/miniTensor/optimizers.h
)

# Create library
add_library(miniTensor STATIC ${MINITENSOR_SOURCES} ${MINITENSOR_HEADERS})

target_include_directories(miniTensor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR}
)

# Create example executable
add_executable(miniTensor_example examples/main.cpp)
target_link_libraries(miniTensor_example miniTensor)

# Set output directory
set_target_properties(miniTensor_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Installation (optional)
install(TARGETS miniTensor
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Print configuration info
message(STATUS "MiniTensor Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Eigen3 Include: ${EIGEN3_INCLUDE_DIR}")
