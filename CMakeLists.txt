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
    libampccl/core/comm_init.cc
    libampccl/core/stream_sync.cc
    libampccl/core/shm_store.cc
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
    libampccl/core/domain_key.h
    libampccl/core/domain.h
    libampccl/core/domain_manager.h
    libampccl/core/shm_store.h
    libampccl/core/comm_init.h
    libampccl/core/planner.h
    libampccl/core/stream_sync.h
    libampccl/core/virtual_collective.h
)

# Create library (name: ampccl, ampccl_nccl, or ampccl_hccl)
add_library(${AMPCCL_TARGET_NAME} SHARED ${AMPCCL_SOURCES} ${AMPCCL_HEADERS})

# No link against libnccl/libhccl at build time; hooks use dlopen/dlsym at runtime.
target_link_libraries(${AMPCCL_TARGET_NAME} PRIVATE
    ${CMAKE_DL_LIBS}  # For dlopen/dlsym
)

# Timer backend: one of CUDA (NCCL) or ACL (HCCL) events, chosen by build target.
# NCCL_ONLY -> AMPCCL_USE_CUDA_TIMER; HCCL_ONLY -> AMPCCL_USE_ACL_TIMER;
# both hooks -> CUDA if found else ACL if ASCEND_HOME else CPU fallback.
if(NCCL_ONLY)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        target_compile_definitions(${AMPCCL_TARGET_NAME} PRIVATE AMPCCL_USE_CUDA_TIMER)
        target_include_directories(${AMPCCL_TARGET_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
        target_link_libraries(${AMPCCL_TARGET_NAME} PRIVATE ${CUDA_LIBRARIES})
        message(STATUS "  Timer: CUDA events (NCCL)")
    else()
        message(STATUS "  Timer: CPU fallback (CUDA not found)")
    endif()
elseif(HCCL_ONLY)
    if(DEFINED ENV{ASCEND_HOME})
        set(ASCEND_HOME "$ENV{ASCEND_HOME}")
    endif()
    if(ASCEND_HOME)
        target_compile_definitions(${AMPCCL_TARGET_NAME} PRIVATE AMPCCL_USE_ACL_TIMER)
        target_include_directories(${AMPCCL_TARGET_NAME} PRIVATE ${ASCEND_HOME}/include)
        target_link_directories(${AMPCCL_TARGET_NAME} PRIVATE ${ASCEND_HOME}/lib64)
        target_link_libraries(${AMPCCL_TARGET_NAME} PRIVATE ascendcl runtime)
        message(STATUS "  Timer: ACL events (HCCL), ASCEND_HOME=${ASCEND_HOME}")
    else()
        message(STATUS "  Timer: CPU fallback (ASCEND_HOME not set)")
    endif()
else()
    # Both hooks: prefer CUDA else ACL else CPU
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        target_compile_definitions(${AMPCCL_TARGET_NAME} PRIVATE AMPCCL_USE_CUDA_TIMER)
        target_include_directories(${AMPCCL_TARGET_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
        target_link_libraries(${AMPCCL_TARGET_NAME} PRIVATE ${CUDA_LIBRARIES})
        message(STATUS "  Timer: CUDA events")
    elseif(DEFINED ENV{ASCEND_HOME})
        set(ASCEND_HOME "$ENV{ASCEND_HOME}")
        target_compile_definitions(${AMPCCL_TARGET_NAME} PRIVATE AMPCCL_USE_ACL_TIMER)
        target_include_directories(${AMPCCL_TARGET_NAME} PRIVATE ${ASCEND_HOME}/include)
        target_link_directories(${AMPCCL_TARGET_NAME} PRIVATE ${ASCEND_HOME}/lib64)
        target_link_libraries(${AMPCCL_TARGET_NAME} PRIVATE ascendcl runtime)
        message(STATUS "  Timer: ACL events")
    else()
        message(STATUS "  Timer: CPU fallback")
    endif()
endif()

# PCIe CCL (pcieccl / PCCL) support
# Set PCIECCL_ROOT to pcieccl project root. Build pcieccl first: make lib DEVICE=ascend
if(ENABLE_PCIE)
    if(DEFINED ENV{PCIECCL_ROOT})
        set(PCIECCL_ROOT "$ENV{PCIECCL_ROOT}")
    endif()
    if(PCIECCL_ROOT)
        set(PCIECCL_INCLUDE_DIR "${PCIECCL_ROOT}/include")
        set(PCIECCL_LIB_DIR "${PCIECCL_ROOT}/build/lib")
        if(EXISTS "${PCIECCL_INCLUDE_DIR}/comm.hpp")
            target_include_directories(${AMPCCL_TARGET_NAME} PRIVATE ${PCIECCL_INCLUDE_DIR})
            target_compile_definitions(${AMPCCL_TARGET_NAME} PRIVATE LOG_LEVEL=1)
            target_link_directories(${AMPCCL_TARGET_NAME} PRIVATE ${PCIECCL_LIB_DIR})
            target_link_libraries(${AMPCCL_TARGET_NAME} PRIVATE pccl numa rt pthread)
            # Ascend (for PCCL built with DEVICE=ascend)
            if(DEFINED ENV{ASCEND_HOME})
                set(ASCEND_HOME "$ENV{ASCEND_HOME}")
                target_link_directories(${AMPCCL_TARGET_NAME} PRIVATE ${ASCEND_HOME}/lib64)
                target_link_libraries(${AMPCCL_TARGET_NAME} PRIVATE ascendcl runtime)
            endif()
            target_compile_definitions(${AMPCCL_TARGET_NAME} PRIVATE AMPCCL_ENABLE_PCIE)
            message(STATUS "  PCIeCCL: ${PCIECCL_ROOT} (PCCL linked)")
        else()
            message(WARNING "PCIECCL_ROOT set but comm.hpp not found; PCIe backend will stub.")
        endif()
    else()
        message(STATUS "  PCIeCCL: not set (set PCIECCL_ROOT to link PCCL)")
    endif()
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
