// HCCL hook for LD_PRELOAD
// This file intercepts HCCL calls and routes them through AMP-CCL

#include "core/virtual_collective.h"
#include "core/domain_manager.h"
#include "core/comm_init.h"
#include "core/stream_sync.h"
#include "common/op_key.h"
#include "common/config.h"
#include <dlfcn.h>
#include <cstring>

// HCCL types (forward declarations if headers not available)
#ifndef HCCL_H
#define HCCL_UNIQUE_ID_BYTES 128
typedef enum {
    HCCL_SUCCESS = 0,
    HCCL_INVALID_PARAM = 1,
    HCCL_INVALID_VALUE = 2
} hcclResult_t;
typedef enum { HCCL_DATA_TYPE_FLOAT = 0, HCCL_DATA_TYPE_FLOAT16, HCCL_DATA_TYPE_INT32 } HcclDataType;
typedef enum { HCCL_REDUCE_SUM = 0, HCCL_REDUCE_MAX, HCCL_REDUCE_MIN } HcclReduceOp;
typedef void* HcclComm;
typedef void* aclrtStream;
typedef struct { char internal[HCCL_UNIQUE_ID_BYTES]; } hcclUniqueId;
#else
#define HCCL_UNIQUE_ID_BYTES sizeof(hcclUniqueId)
#endif

// Forward declarations for original HCCL functions
typedef hcclResult_t (*hcclGetUniqueId_t)(hcclUniqueId* uniqueId);
typedef hcclResult_t (*hcclCommInitRank_t)(HcclComm* comm, unsigned int nranks, hcclUniqueId commId, unsigned int rank);
typedef hcclResult_t (*hcclCommDestroy_t)(HcclComm comm);

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

// ACL runtime (for stream sync)
typedef int (*aclrtSynchronizeStream_t)(aclrtStream stream);

// Function pointers to original HCCL functions
static hcclGetUniqueId_t orig_hcclGetUniqueId = nullptr;
static hcclCommInitRank_t orig_hcclCommInitRank = nullptr;
static hcclCommDestroy_t orig_hcclCommDestroy = nullptr;
static hcclAllReduce_t orig_hcclAllReduce = nullptr;
static hcclAllGather_t orig_hcclAllGather = nullptr;
static hcclReduceScatter_t orig_hcclReduceScatter = nullptr;
static hcclBroadcast_t orig_hcclBroadcast = nullptr;
static aclrtSynchronizeStream_t orig_aclrtSynchronizeStream = nullptr;

// Load original HCCL functions
static void LoadOriginalFunctions() {
    static bool loaded = false;
    if (loaded) return;

    void* handle = dlopen("libhccl.so", RTLD_LAZY);
    if (!handle) {
        handle = dlopen("libhccl.so.1", RTLD_LAZY);
    }

    if (handle) {
        orig_hcclGetUniqueId = (hcclGetUniqueId_t)dlsym(handle, "HcclGetUniqueId");
        orig_hcclCommInitRank = (hcclCommInitRank_t)dlsym(handle, "HcclCommInitRank");
        orig_hcclCommDestroy = (hcclCommDestroy_t)dlsym(handle, "HcclCommDestroy");
        orig_hcclAllReduce = (hcclAllReduce_t)dlsym(handle, "HcclAllReduce");
        orig_hcclAllGather = (hcclAllGather_t)dlsym(handle, "HcclAllGather");
        orig_hcclReduceScatter = (hcclReduceScatter_t)dlsym(handle, "HcclReduceScatter");
        orig_hcclBroadcast = (hcclBroadcast_t)dlsym(handle, "HcclBroadcast");
    }

    // ACL runtime for aclrtSynchronizeStream (may be in same lib or libascendcl/libacl)
    if (!orig_aclrtSynchronizeStream) {
        void* acl_handle = dlopen("libascendcl.so", RTLD_LAZY);
        if (!acl_handle) {
            acl_handle = dlopen("libacl.so", RTLD_LAZY);
        }
        if (acl_handle) {
            orig_aclrtSynchronizeStream = (aclrtSynchronizeStream_t)dlsym(acl_handle, "aclrtSynchronizeStream");
        }
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

// Look up domain by raw HCCL communicator (registered at CommInit).
static ampccl::CommDomain* GetDomainByRawComm(HcclComm comm) {
    return ampccl::DomainManager::GetInstance().GetDomainByRawComm(comm);
}

// Hooked HCCL functions
extern "C" {

hcclResult_t HcclGetUniqueId(hcclUniqueId* uniqueId) {
    LoadOriginalFunctions();
    if (orig_hcclGetUniqueId) {
        return orig_hcclGetUniqueId(uniqueId);
    }
    return HCCL_INVALID_PARAM;
}

hcclResult_t HcclCommInitRank(HcclComm* comm, unsigned int nranks, hcclUniqueId commId, unsigned int rank) {
    LoadOriginalFunctions();
    if (!orig_hcclCommInitRank) {
        return HCCL_INVALID_PARAM;
    }
    hcclResult_t ret = orig_hcclCommInitRank(comm, nranks, commId, rank);
    if (ret != HCCL_SUCCESS || comm == nullptr || *comm == nullptr) {
        return ret;
    }
    if (!ampccl::Config::IsAdaptiveEnabled()) {
        return ret;
    }
    ampccl::CommDomainKey key = ampccl::BuildKeyFromHcclInit(
        static_cast<int>(nranks), &commId, HCCL_UNIQUE_ID_BYTES, static_cast<int>(rank));
    ampccl::DomainManager::GetInstance().RegisterRawComm(*comm, key);
    ampccl::CommDomain* domain = ampccl::DomainManager::GetInstance().GetDomainByRawComm(*comm);
    if (domain) {
        ampccl::InitPCIeForDomain(domain, static_cast<int>(rank), static_cast<int>(nranks));
    }
    return ret;
}

hcclResult_t HcclCommDestroy(HcclComm comm) {
    LoadOriginalFunctions();
    if (ampccl::Config::IsAdaptiveEnabled()) {
        ampccl::DomainManager::GetInstance().UnregisterRawComm(comm);
    }
    if (orig_hcclCommDestroy) {
        return orig_hcclCommDestroy(comm);
    }
    return HCCL_INVALID_PARAM;
}

hcclResult_t HcclAllReduce(
    const void* sendbuff, void* recvbuff, unsigned long count,
    HcclDataType datatype, HcclReduceOp op, HcclComm comm, aclrtStream stream) {

    LoadOriginalFunctions();
    if (!ampccl::Config::IsAdaptiveEnabled() && orig_hcclAllReduce) {
        return orig_hcclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
    }

    ampccl::CommDomain* domain = GetDomainByRawComm(comm);
    if (!domain) {
        if (orig_hcclAllReduce) {
            return orig_hcclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
        }
        return HCCL_INVALID_PARAM;
    }

    ampccl::BackendResult result = ampccl::VirtualCollective::AllReduce(
        domain, sendbuff, recvbuff, count,
        static_cast<int>(datatype), static_cast<int>(op), comm, stream);

    return (result == ampccl::BackendResult::Success) ? HCCL_SUCCESS : HCCL_INVALID_PARAM;
}

hcclResult_t HcclAllGather(
    const void* sendbuff, void* recvbuff, unsigned long sendcount,
    HcclDataType datatype, HcclComm comm, aclrtStream stream) {

    LoadOriginalFunctions();
    if (!ampccl::Config::IsAdaptiveEnabled() && orig_hcclAllGather) {
        return orig_hcclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    }

    ampccl::CommDomain* domain = GetDomainByRawComm(comm);
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
    if (orig_hcclReduceScatter) {
        return orig_hcclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
    }
    return HCCL_INVALID_PARAM;
}

hcclResult_t HcclBroadcast(
    const void* sendbuff, void* recvbuff, unsigned long count,
    HcclDataType datatype, unsigned int root, HcclComm comm, aclrtStream stream) {

    LoadOriginalFunctions();
    if (orig_hcclBroadcast) {
        return orig_hcclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
    }
    return HCCL_INVALID_PARAM;
}

int aclrtSynchronizeStream(aclrtStream stream) {
    LoadOriginalFunctions();
    if (orig_aclrtSynchronizeStream) {
        int ret = orig_aclrtSynchronizeStream(stream);
        if (ret == 0 && ampccl::Config::IsAdaptiveEnabled()) {
            ampccl::OnStreamSynchronized(stream);
        }
        return ret;
    }
    return -1;
}

}  // extern "C"
