#ifndef AMPCCL_BACKEND_PCIE_BACKEND_H_
#define AMPCCL_BACKEND_PCIE_BACKEND_H_

#include "backend_base.h"

namespace ampccl {

// PCIe backend tag
struct PCIeBackend {};

// Specialization for PCIe backend
// This will call vendor-provided PCIe CCL APIs (to be included later)
template<>
class BackendBase<PCIeBackend> {
public:
    static BackendResult AllReduce(
        const void* sendbuff,
        void* recvbuff,
        size_t count,
        int datatype,
        int op,
        void* comm,
        void* stream
    );

    static BackendResult AllGather(
        const void* sendbuff,
        void* recvbuff,
        size_t sendcount,
        int datatype,
        void* comm,
        void* stream
    );

    static BackendResult ReduceScatter(
        const void* sendbuff,
        void* recvbuff,
        size_t recvcount,
        int datatype,
        int op,
        void* comm,
        void* stream
    );

    static BackendResult Broadcast(
        const void* sendbuff,
        void* recvbuff,
        size_t count,
        int datatype,
        int root,
        void* comm,
        void* stream
    );
};

// Type alias for convenience
using PCIeBackendImpl = BackendBase<PCIeBackend>;

}  // namespace ampccl

#endif  // AMPCCL_BACKEND_PCIE_BACKEND_H_
