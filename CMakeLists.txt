cmake_minimum_required(VERSION 3.12)
project(Adaptive-CCL VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_SHARED_LIBS "Build shared library" ON)
option(ENABLE_NCCL "Include NCCL hook in build" ON)
option(ENABLE_HCCL "Include HCCL hook in build" ON)
option(ENABLE_PCIE "Enable PCIe backend" ON)
# Build only one hook -> smaller .so (libampccl_nccl.so or libampccl_hccl.so)
option(NCCL_ONLY "Build only NCCL hook (output: libampccl_nccl.so)" OFF)
option(HCCL_ONLY "Build only HCCL hook (output: libampccl_hccl.so)" OFF)

# Include directories
set(AMPCCL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libampccl")
include_directories(${AMPCCL_INCLUDE_DIR})

# Core sources (no hooks)
set(AMPCCL_CORE_SOURCES
    libampccl/backend/fast_backend.cc
    libampccl/backend/pcie_backend.cc
)

# Hook sources (NCCL_ONLY takes precedence if both set)
if(NCCL_ONLY)
    set(AMPCCL_HOOK_SOURCES libampccl/hook/nccl_hook.cc)
    set(AMPCCL_TARGET_NAME ampccl_nccl)
elseif(HCCL_ONLY)
    set(AMPCCL_HOOK_SOURCES libampccl/hook/hccl_hook.cc)
    set(AMPCCL_TARGET_NAME ampccl_hccl)
else()
    set(AMPCCL_HOOK_SOURCES
        libampccl/hook/nccl_hook.cc
        libampccl/hook/hccl_hook.cc
    )
    set(AMPCCL_TARGET_NAME ampccl)
endif()

set(AMPCCL_SOURCES ${AMPCCL_CORE_SOURCES} ${AMPCCL_HOOK_SOURCES})

# Header files (for IDE support)
set(AMPCCL_HEADERS
    libampccl/common/op_key.h
    libampccl/common/config.h
    libampccl/common/log.h
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
    libampccl/core/comm_init.h
    libampccl/core/planner.h
    libampccl/core/virtual_collective.h
)

# Create library (name: ampccl, ampccl_nccl, or ampccl_hccl)
add_library(${AMPCCL_TARGET_NAME} SHARED ${AMPCCL_SOURCES} ${AMPCCL_HEADERS})

# No link against libnccl/libhccl at build time; hooks use dlopen/dlsym at runtime.
target_link_libraries(${AMPCCL_TARGET_NAME} PRIVATE
    ${CMAKE_DL_LIBS}  # For dlopen/dlsym
)

# CUDA support (optional, for timer.h)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    target_include_directories(${AMPCCL_TARGET_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${AMPCCL_TARGET_NAME} PRIVATE ${CUDA_LIBRARIES})
    target_compile_definitions(${AMPCCL_TARGET_NAME} PRIVATE __CUDACC__)
endif()

# PCIe CCL support (to be linked when available)
if(ENABLE_PCIE)
    target_compile_definitions(${AMPCCL_TARGET_NAME} PRIVATE AMPCCL_ENABLE_PCIE)
endif()

# Installation
install(TARGETS ${AMPCCL_TARGET_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY libampccl/
    DESTINATION include/libampccl
    FILES_MATCHING PATTERN "*.h"
)

# Export for use in other projects
set(AMPCCL_INCLUDE_DIRS ${AMPCCL_INCLUDE_DIR})
set(AMPCCL_LIBRARIES ${AMPCCL_TARGET_NAME})

# Print configuration
message(STATUS "Adaptive-CCL Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Output: lib${AMPCCL_TARGET_NAME}.so")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  NCCL_ONLY: ${NCCL_ONLY}")
message(STATUS "  HCCL_ONLY: ${HCCL_ONLY}")
message(STATUS "  PCIe support: ${ENABLE_PCIE}")
