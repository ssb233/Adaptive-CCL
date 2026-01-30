#ifndef AMPCCL_CORE_COMM_INIT_H_
#define AMPCCL_CORE_COMM_INIT_H_

#include "domain.h"
#include <cstddef>
#include <vector>

namespace ampccl {

// Build our Comm identity (CommDomainKey) from NCCL init parameters.
// Same (nranks, commId, rank set) across ranks yields the same key, so
// dividing-param table is keyed by our Comm and can be shared/reused.
inline CommDomainKey BuildKeyFromNcclInit(int nranks,
                                         const void* comm_id_bytes,
                                         size_t comm_id_len,
                                         int rank) {
    CommDomainKey key;
    key.world_size = nranks;
    key.ranks.resize(static_cast<size_t>(nranks));
    for (int i = 0; i < nranks; ++i) {
        key.ranks[static_cast<size_t>(i)] = i;
    }
    // Topology / clique identity from NCCL unique id
    key.topology_hash = 0;
    if (comm_id_bytes && comm_id_len > 0) {
        const unsigned char* p = static_cast<const unsigned char*>(comm_id_bytes);
        for (size_t i = 0; i < comm_id_len; ++i) {
            key.topology_hash = key.topology_hash * 131 + static_cast<uint64_t>(p[i]);
        }
    }
    return key;
}

// Build our Comm identity from HCCL init parameters.
inline CommDomainKey BuildKeyFromHcclInit(int nranks,
                                          const void* comm_id_bytes,
                                          size_t comm_id_len,
                                          int rank) {
    return BuildKeyFromNcclInit(nranks, comm_id_bytes, comm_id_len, rank);
}

// PCIe communicator / resource init for this domain. Reserved for later.
// Called after fast backend (NCCL/HCCL) comm is created.
inline void InitPCIeForDomain(CommDomain* domain) {
    (void)domain;
    // TODO: Initialize PCIe CCL context for this domain when API is available.
}

}  // namespace ampccl

#endif  // AMPCCL_CORE_COMM_INIT_H_
