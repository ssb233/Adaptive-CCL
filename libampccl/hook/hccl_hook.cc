// HCCL hook for LD_PRELOAD
// This file intercepts HCCL calls and routes them through AMP-CCL

#include "core/virtual_collective.h"
#include "core/domain_manager.h"
#include "common/op_key.h"
#include <dlfcn.h>
#include <cstring>

// HCCL types (forward declarations if headers not available)
#ifndef HCCL_H
typedef enum {
    HCCL_SUCCESS = 0,
    HCCL_INVALID_PARAM = 1,
    HCCL_INVALID_VALUE = 2
} hcclResult_t;
typedef enum { HCCL_DATA_TYPE_FLOAT = 0, HCCL_DATA_TYPE_FLOAT16, HCCL_DATA_TYPE_INT32 } HcclDataType;
typedef enum { HCCL_REDUCE_SUM = 0, HCCL_REDUCE_MAX, HCCL_REDUCE_MIN } HcclReduceOp;
typedef void* HcclComm;
typedef void* aclrtStream;
#endif

// Forward declarations for original HCCL functions
// Note: HCCL uses hcclResult_t return type
typedef hcclResult_t (*hcclAllReduce_t)(
    const void* sendbuff, void* recvbuff, unsigned long count,
    HcclDataType datatype, HcclReduceOp op, HcclComm comm, aclrtStream stream);

typedef hcclResult_t (*hcclAllGather_t)(
    const void* sendbuff, void* recvbuff, unsigned long sendcount,
    HcclDataType datatype, HcclComm comm, aclrtStream stream);

typedef hcclResult_t (*hcclReduceScatter_t)(
    const void* sendbuff, void* recvbuff, unsigned long recvcount,
    HcclDataType datatype, HcclReduceOp op, HcclComm comm, aclrtStream stream);

typedef hcclResult_t (*hcclBroadcast_t)(
    const void* sendbuff, void* recvbuff, unsigned long count,
    HcclDataType datatype, unsigned int root, HcclComm comm, aclrtStream stream);

// Function pointers to original HCCL functions
static hcclAllReduce_t orig_hcclAllReduce = nullptr;
static hcclAllGather_t orig_hcclAllGather = nullptr;
static hcclReduceScatter_t orig_hcclReduceScatter = nullptr;
static hcclBroadcast_t orig_hcclBroadcast = nullptr;

// Load original HCCL functions
static void LoadOriginalFunctions() {
    static bool loaded = false;
    if (loaded) return;

    void* handle = dlopen("libhccl.so", RTLD_LAZY);
    if (!handle) {
        handle = dlopen("libhccl.so.1", RTLD_LAZY);
    }

    if (handle) {
        orig_hcclAllReduce = (hcclAllReduce_t)dlsym(handle, "HcclAllReduce");
        orig_hcclAllGather = (hcclAllGather_t)dlsym(handle, "HcclAllGather");
        orig_hcclReduceScatter = (hcclReduceScatter_t)dlsym(handle, "HcclReduceScatter");
        orig_hcclBroadcast = (hcclBroadcast_t)dlsym(handle, "HcclBroadcast");
    }

    loaded = true;
}

// Helper to convert HCCL result to BackendResult
static ampccl::BackendResult ConvertHCCLResult(hcclResult_t hccl_result) {
    if (hccl_result == HCCL_SUCCESS) {
        return ampccl::BackendResult::Success;
    }
    return ampccl::BackendResult::UnhandledError;
}

// Helper to get or create domain for HCCL communicator
static ampccl::CommDomain* GetDomainForComm(HcclComm comm) {
    // Create domain key from communicator
    ampccl::CommDomainKey key;
    key.world_size = 0;  // Should be extracted from comm
    key.topology_hash = 0;  // Should be computed from topology
    key.ranks.clear();  // Should be extracted from comm

    return ampccl::DomainManager::GetInstance().GetOrCreateDomain(comm, key);
}

// Hooked HCCL functions
extern "C" {

hcclResult_t HcclAllReduce(
    const void* sendbuff, void* recvbuff, unsigned long count,
    HcclDataType datatype, HcclReduceOp op, HcclComm comm, aclrtStream stream) {

    LoadOriginalFunctions();

    // Get domain for this communicator
    ampccl::CommDomain* domain = GetDomainForComm(comm);
    if (!domain) {
        // Fallback to original if domain creation fails
        if (orig_hcclAllReduce) {
            return orig_hcclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
        }
        return HCCL_INVALID_PARAM;
    }

    // Route through virtual collective
    ampccl::BackendResult result = ampccl::VirtualCollective::AllReduce(
        domain, sendbuff, recvbuff, count,
        static_cast<int>(datatype), static_cast<int>(op), comm, stream);

    return (result == ampccl::BackendResult::Success) ? HCCL_SUCCESS : HCCL_INVALID_PARAM;
}

hcclResult_t HcclAllGather(
    const void* sendbuff, void* recvbuff, unsigned long sendcount,
    HcclDataType datatype, HcclComm comm, aclrtStream stream) {

    LoadOriginalFunctions();

    ampccl::CommDomain* domain = GetDomainForComm(comm);
    if (!domain) {
        if (orig_hcclAllGather) {
            return orig_hcclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
        }
        return HCCL_INVALID_PARAM;
    }

    ampccl::BackendResult result = ampccl::VirtualCollective::AllGather(
        domain, sendbuff, recvbuff, sendcount,
        static_cast<int>(datatype), comm, stream);

    return (result == ampccl::BackendResult::Success) ? HCCL_SUCCESS : HCCL_INVALID_PARAM;
}

hcclResult_t HcclReduceScatter(
    const void* sendbuff, void* recvbuff, unsigned long recvcount,
    HcclDataType datatype, HcclReduceOp op, HcclComm comm, aclrtStream stream) {

    LoadOriginalFunctions();

    // Similar implementation...
    if (orig_hcclReduceScatter) {
        return orig_hcclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
    }
    return HCCL_INVALID_PARAM;
}

hcclResult_t HcclBroadcast(
    const void* sendbuff, void* recvbuff, unsigned long count,
    HcclDataType datatype, unsigned int root, HcclComm comm, aclrtStream stream) {

    LoadOriginalFunctions();

    // Similar implementation...
    if (orig_hcclBroadcast) {
        return orig_hcclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
    }
    return HCCL_INVALID_PARAM;
}

}  // extern "C"
