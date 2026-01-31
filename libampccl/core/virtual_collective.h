#ifndef AMPCCL_CORE_VIRTUAL_COLLECTIVE_H_
#define AMPCCL_CORE_VIRTUAL_COLLECTIVE_H_

#include "domain.h"
#include "domain_manager.h"
#include "planner.h"
#include "common/op_key.h"
#include "backend/fast_backend.h"
#include "backend/pcie_backend.h"
#include "telemetry/timer.h"
#include "telemetry/stats.h"
#include "common/config.h"
#include "common/log.h"
#include <cstddef>
#include <cstring>

namespace ampccl {

// Virtual collective layer - the heart of the system
class VirtualCollective {
public:
    // AllReduce operation
    static BackendResult AllReduce(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t count,
        int datatype,
        int op,
        void* comm,
        void* stream
    ) {
        // 1. Build OpKey
        OpKey op_key;
        op_key.op = CollectiveType::AllReduce;
        op_key.bytes = count * GetDataTypeSize(datatype);
        op_key.datatype = datatype;

        // 2. ParamCache lookup
        ParamValue param = domain->param_cache.Lookup(op_key);

        // 3. Controller suggests alpha
        double alpha = domain->controller->SuggestAlpha(op_key, domain->param_cache);

        // 4. Planner builds split plan
        Plan plan = Planner::CreatePlan(op_key.bytes, alpha, param.use_pcie);

        // 5. Launch fast + PCIe backend
        ExecStat stat;
        Timer timer;

        AMPCCL_LOG(INFO, "AllReduce before CCL: op=AllReduce bytes=%zu datatype=%d alpha=%.3f use_pcie=%d fast_bytes=%zu pcie_bytes=%zu",
                   op_key.bytes, datatype, alpha, plan.use_pcie ? 1 : 0, plan.fast_bytes, plan.pcie_bytes);

        if (plan.use_pcie && plan.pcie_bytes > 0) {
            // Split buffer and launch both backends
            size_t fast_offset = 0;
            size_t pcie_offset = plan.fast_bytes;
            size_t elem_size = GetDataTypeSize(datatype);

            // Fast backend
            if (plan.fast_bytes > 0) {
                timer.Start();
                void* fast_send = const_cast<void*>(sendbuff);
                void* fast_recv = recvbuff;

                BackendResult fast_result = FastBackendImpl::AllReduce(
                    fast_send, fast_recv, plan.fast_bytes / elem_size,
                    datatype, op, comm, stream);

                timer.Stop();
                stat.fast_time = timer.ElapsedSeconds();
                stat.fast_bytes = plan.fast_bytes;
                stat.fast_success = (fast_result == BackendResult::Success);
            }

            // PCIe backend (offset buffer) — uses domain's pcie_comm
            if (plan.pcie_bytes > 0) {
                timer.Start();
                const char* pcie_send = static_cast<const char*>(sendbuff) + pcie_offset;
                char* pcie_recv = static_cast<char*>(recvbuff) + pcie_offset;

                BackendResult pcie_result = PCIeBackendImpl::AllReduce(
                    domain, pcie_send, pcie_recv, plan.pcie_bytes / elem_size,
                    datatype, op, stream);

                timer.Stop();
                stat.pcie_time = timer.ElapsedSeconds();
                stat.pcie_bytes = plan.pcie_bytes;
                stat.pcie_success = (pcie_result == BackendResult::Success);
            }
        } else {
            // Fast-only path
            timer.Start();
            BackendResult result = FastBackendImpl::AllReduce(
                sendbuff, recvbuff, count, datatype, op, comm, stream);
            timer.Stop();
            stat.fast_time = timer.ElapsedSeconds();
            stat.fast_bytes = op_key.bytes;
            stat.fast_success = (result == BackendResult::Success);
            stat.pcie_time = 0.0;
            stat.pcie_bytes = 0;
            stat.pcie_success = true;
        }

        // 6. Wait (handled by backend synchronization)

        // 7. Measure times (already done above)

        // 8. Controller.update()
        // 9. Cache.update()
        domain->controller->Update(op_key, stat, domain->param_cache);

        AMPCCL_LOG(INFO, "AllReduce after CCL: fast_time=%.6fs pcie_time=%.6fs fast_bytes=%zu pcie_bytes=%zu fast_ok=%d pcie_ok=%d (plan: alpha=%.3f fast_bytes=%zu pcie_bytes=%zu)",
                   stat.fast_time, stat.pcie_time, stat.fast_bytes, stat.pcie_bytes,
                   stat.fast_success ? 1 : 0, stat.pcie_success ? 1 : 0,
                   alpha, plan.fast_bytes, plan.pcie_bytes);

        return stat.fast_success && stat.pcie_success ?
               BackendResult::Success : BackendResult::UnhandledError;
    }

    // Similar implementations for other collectives...
    static BackendResult AllGather(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t sendcount,
        int datatype,
        void* comm,
        void* stream
    ) {
        // Similar to AllReduce but for AllGather
        OpKey op_key;
        op_key.op = CollectiveType::AllGather;
        op_key.bytes = sendcount * GetDataTypeSize(datatype);
        op_key.datatype = datatype;

        ParamValue param = domain->param_cache.Lookup(op_key);
        double alpha = domain->controller->SuggestAlpha(op_key, domain->param_cache);
        Plan plan = Planner::CreatePlan(op_key.bytes, alpha, param.use_pcie);

        AMPCCL_LOG(INFO, "AllGather before CCL: bytes=%zu datatype=%d alpha=%.3f use_pcie=%d fast_bytes=%zu pcie_bytes=%zu",
                   op_key.bytes, datatype, alpha, plan.use_pcie ? 1 : 0, plan.fast_bytes, plan.pcie_bytes);

        ExecStat stat;
        Timer timer;

        if (plan.use_pcie && plan.pcie_bytes > 0) {
            size_t fast_offset = 0;
            size_t pcie_offset = plan.fast_bytes;
            size_t elem_size = GetDataTypeSize(datatype);

            if (plan.fast_bytes > 0) {
                timer.Start();
                BackendResult fast_result = FastBackendImpl::AllGather(
                    sendbuff, recvbuff, plan.fast_bytes / elem_size, datatype, comm, stream);
                timer.Stop();
                stat.fast_time = timer.ElapsedSeconds();
                stat.fast_bytes = plan.fast_bytes;
                stat.fast_success = (fast_result == BackendResult::Success);
            }
            if (plan.pcie_bytes > 0) {
                timer.Start();
                const char* pcie_send = static_cast<const char*>(sendbuff) + pcie_offset;
                char* pcie_recv = static_cast<char*>(recvbuff) + pcie_offset;
                // PCCL 2-rank AllGather: recvbuff holds 2 chunks; per-chunk elements = pcie_bytes/(2*elem_size)
                size_t pcie_chunk_elems = plan.pcie_bytes / (2 * elem_size);
                BackendResult pcie_result = PCIeBackendImpl::AllGather(
                    domain, pcie_send, pcie_recv, pcie_chunk_elems, datatype, stream);
                timer.Stop();
                stat.pcie_time = timer.ElapsedSeconds();
                stat.pcie_bytes = plan.pcie_bytes;
                stat.pcie_success = (pcie_result == BackendResult::Success);
            }
        } else {
            timer.Start();
            BackendResult result = FastBackendImpl::AllGather(
                sendbuff, recvbuff, sendcount, datatype, comm, stream);
            timer.Stop();
            stat.fast_time = timer.ElapsedSeconds();
            stat.fast_bytes = op_key.bytes;
            stat.fast_success = (result == BackendResult::Success);
        }

        domain->controller->Update(op_key, stat, domain->param_cache);

        AMPCCL_LOG(INFO, "AllGather after CCL: fast_time=%.6fs pcie_time=%.6fs fast_bytes=%zu fast_ok=%d (plan: alpha=%.3f fast_bytes=%zu pcie_bytes=%zu)",
                   stat.fast_time, stat.pcie_time, stat.fast_bytes, stat.fast_success ? 1 : 0,
                   alpha, plan.fast_bytes, plan.pcie_bytes);

        return stat.fast_success ? BackendResult::Success : BackendResult::UnhandledError;
    }

private:
    static size_t GetDataTypeSize(int datatype) {
        // Map NCCL/HCCL datatype to size
        // This is a simplified version - actual implementation should handle all types
        switch (datatype) {
            case 0: return 4;   // float32
            case 1: return 8;   // float64
            case 2: return 2;   // float16
            case 3: return 4;   // int32
            case 4: return 8;   // int64
            default: return 4;
        }
    }
};

}  // namespace ampccl

#endif  // AMPCCL_CORE_VIRTUAL_COLLECTIVE_H_
