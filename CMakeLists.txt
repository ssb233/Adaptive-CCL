cmake_minimum_required(VERSION 3.12)
project(Adaptive-CCL VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_SHARED_LIBS "Build shared library" ON)
option(ENABLE_NCCL "Enable NCCL support" ON)
option(ENABLE_HCCL "Enable HCCL support" ON)
option(ENABLE_PCIE "Enable PCIe backend" ON)

# Include directories
set(AMPCCL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libampccl")
include_directories(${AMPCCL_INCLUDE_DIR})

# Source files
set(AMPCCL_SOURCES
    libampccl/backend/fast_backend.cc
    libampccl/backend/pcie_backend.cc
    libampccl/hook/nccl_hook.cc
    libampccl/hook/hccl_hook.cc
)

# Header files (for IDE support)
set(AMPCCL_HEADERS
    libampccl/common/op_key.h
    libampccl/common/config.h
    libampccl/telemetry/timer.h
    libampccl/telemetry/stats.h
    libampccl/cache/param_cache.h
    libampccl/backend/backend_base.h
    libampccl/backend/fast_backend.h
    libampccl/backend/pcie_backend.h
    libampccl/controller/algo_base.h
    libampccl/controller/algo_tcp.h
    libampccl/controller/algo_dcqcn.h
    libampccl/controller/algo_factory.h
    libampccl/controller/controller.h
    libampccl/core/domain.h
    libampccl/core/domain_manager.h
    libampccl/core/planner.h
    libampccl/core/virtual_collective.h
)

# Create library
add_library(ampccl ${AMPCCL_SOURCES} ${AMPCCL_HEADERS})

# Link libraries
target_link_libraries(ampccl PRIVATE
    ${CMAKE_DL_LIBS}  # For dlopen/dlsym
)

# CUDA support (optional)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    target_include_directories(ampccl PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(ampccl PRIVATE ${CUDA_LIBRARIES})
    target_compile_definitions(ampccl PRIVATE __CUDACC__)
endif()

# NCCL support (optional)
if(ENABLE_NCCL)
    find_library(NCCL_LIBRARY
        NAMES nccl
        PATHS /usr/local/lib /usr/lib
        PATH_SUFFIXES lib lib64
    )
    if(NCCL_LIBRARY)
        target_link_libraries(ampccl PRIVATE ${NCCL_LIBRARY})
        target_compile_definitions(ampccl PRIVATE AMPCCL_ENABLE_NCCL)
    endif()
endif()

# HCCL support (optional)
if(ENABLE_HCCL)
    find_library(HCCL_LIBRARY
        NAMES hccl
        PATHS /usr/local/lib /usr/lib
        PATH_SUFFIXES lib lib64
    )
    if(HCCL_LIBRARY)
        target_link_libraries(ampccl PRIVATE ${HCCL_LIBRARY})
        target_compile_definitions(ampccl PRIVATE AMPCCL_ENABLE_HCCL)
    endif()
endif()

# PCIe CCL support (to be linked when available)
if(ENABLE_PCIE)
    # TODO: Add PCIe CCL library when available
    # find_library(PCIECCL_LIBRARY ...)
    # target_link_libraries(ampccl PRIVATE ${PCIECCL_LIBRARY})
    target_compile_definitions(ampccl PRIVATE AMPCCL_ENABLE_PCIE)
endif()

# Installation
install(TARGETS ampccl
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY libampccl/
    DESTINATION include/libampccl
    FILES_MATCHING PATTERN "*.h"
)

# Export for use in other projects
set(AMPCCL_INCLUDE_DIRS ${AMPCCL_INCLUDE_DIR})
set(AMPCCL_LIBRARIES ampccl)

# Print configuration
message(STATUS "Adaptive-CCL Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Shared library: ${BUILD_SHARED_LIBS}")
message(STATUS "  NCCL support: ${ENABLE_NCCL}")
message(STATUS "  HCCL support: ${ENABLE_HCCL}")
message(STATUS "  PCIe support: ${ENABLE_PCIE}")
