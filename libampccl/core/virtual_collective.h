#ifndef AMPCCL_CORE_VIRTUAL_COLLECTIVE_H_
#define AMPCCL_CORE_VIRTUAL_COLLECTIVE_H_

#include "domain.h"
#include "domain_manager.h"
#include "planner.h"
#include "common/op_key.h"
#include "backend/fast_backend.h"
#include "backend/pcie_backend.h"
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

        domain->EnsureShmAttached();
        ShmParamStore* shm = domain->shm_store();
        if (shm->IsAttached() && shm->IsRank0()) {
            ExecStat global_stat;
            OpKey agg_op_key;
            if (shm->ReadAllStatsAndAggregate(&global_stat, &agg_op_key) && domain->controller) {
                domain->controller->Update(agg_op_key, global_stat, domain->param_cache);
                shm->WriteParams(domain->param_cache);
            }
        }
        if (shm->IsAttached()) {
            shm->ReadParams(&domain->param_cache);
        }

        // 2. ParamCache lookup
        ParamValue param = domain->param_cache.Lookup(op_key);

        // 3. Controller suggests alpha
        double alpha = domain->controller->SuggestAlpha(op_key, domain->param_cache);

        // 4. Planner builds split plan
        Plan plan = Planner::CreatePlan(op_key.bytes, alpha, param.use_pcie);

        // 5. Launch fast + PCIe backend: record events only, no sync; pending consumed at SynchronizeStream.
        AMPCCL_LOG(INFO, "AllReduce before CCL: op=AllReduce bytes=%zu datatype=%d alpha=%.3f use_pcie=%d fast_bytes=%zu pcie_bytes=%zu",
                   op_key.bytes, datatype, alpha, plan.use_pcie ? 1 : 0, plan.fast_bytes, plan.pcie_bytes);

        bool fast_ok = true;
        bool pcie_ok = true;
        void* pcie_stream = domain->pcie_stream();

        if (plan.use_pcie && plan.pcie_bytes > 0 && pcie_stream) {
            size_t fast_offset = 0;
            size_t pcie_offset = plan.fast_bytes;
            size_t elem_size = GetDataTypeSize(datatype);

            if (plan.fast_bytes > 0) {
                domain->timer_fast().Start(stream);
                void* fast_send = const_cast<void*>(sendbuff);
                void* fast_recv = recvbuff;
                BackendResult fast_result = FastBackendImpl::AllReduce(
                    fast_send, fast_recv, plan.fast_bytes / elem_size,
                    datatype, op, comm, stream);
                domain->timer_fast().Stop(stream);
                fast_ok = (fast_result == BackendResult::Success);
            }
            if (plan.pcie_bytes > 0) {
                domain->timer_pcie().Start(pcie_stream);
                const char* pcie_send = static_cast<const char*>(sendbuff) + pcie_offset;
                char* pcie_recv = static_cast<char*>(recvbuff) + pcie_offset;
                BackendResult pcie_result = PCIeBackendImpl::AllReduce(
                    domain, pcie_send, pcie_recv, plan.pcie_bytes / elem_size,
                    datatype, op, pcie_stream);
                domain->timer_pcie().Stop(pcie_stream);
                pcie_ok = (pcie_result == BackendResult::Success);
            }
        } else {
            domain->timer_fast().Start(stream);
            BackendResult result = FastBackendImpl::AllReduce(
                sendbuff, recvbuff, count, datatype, op, comm, stream);
            domain->timer_fast().Stop(stream);
            fast_ok = (result == BackendResult::Success);
        }

        DomainManager::GetInstance().RegisterStreamPending(
            stream, domain, op_key, plan, fast_ok, pcie_ok);

        return (fast_ok && pcie_ok) ? BackendResult::Success : BackendResult::UnhandledError;
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

        domain->EnsureShmAttached();
        ShmParamStore* shm = domain->shm_store();
        if (shm->IsAttached() && shm->IsRank0()) {
            ExecStat global_stat;
            OpKey agg_op_key;
            if (shm->ReadAllStatsAndAggregate(&global_stat, &agg_op_key) && domain->controller) {
                domain->controller->Update(agg_op_key, global_stat, domain->param_cache);
                shm->WriteParams(domain->param_cache);
            }
        }
        if (shm->IsAttached()) {
            shm->ReadParams(&domain->param_cache);
        }

        ParamValue param = domain->param_cache.Lookup(op_key);
        double alpha = domain->controller->SuggestAlpha(op_key, domain->param_cache);
        Plan plan = Planner::CreatePlan(op_key.bytes, alpha, param.use_pcie);

        AMPCCL_LOG(INFO, "AllGather before CCL: bytes=%zu datatype=%d alpha=%.3f use_pcie=%d fast_bytes=%zu pcie_bytes=%zu",
                   op_key.bytes, datatype, alpha, plan.use_pcie ? 1 : 0, plan.fast_bytes, plan.pcie_bytes);

        bool fast_ok = true;
        bool pcie_ok = true;
        void* pcie_stream = domain->pcie_stream();

        if (plan.use_pcie && plan.pcie_bytes > 0 && pcie_stream) {
            size_t fast_offset = 0;
            size_t pcie_offset = plan.fast_bytes;
            size_t elem_size = GetDataTypeSize(datatype);

            if (plan.fast_bytes > 0) {
                domain->timer_fast().Start(stream);
                BackendResult fast_result = FastBackendImpl::AllGather(
                    sendbuff, recvbuff, plan.fast_bytes / elem_size, datatype, comm, stream);
                domain->timer_fast().Stop(stream);
                fast_ok = (fast_result == BackendResult::Success);
            }
            if (plan.pcie_bytes > 0) {
                domain->timer_pcie().Start(pcie_stream);
                const char* pcie_send = static_cast<const char*>(sendbuff) + pcie_offset;
                char* pcie_recv = static_cast<char*>(recvbuff) + pcie_offset;
                size_t pcie_chunk_elems = plan.pcie_bytes / (2 * elem_size);
                BackendResult pcie_result = PCIeBackendImpl::AllGather(
                    domain, pcie_send, pcie_recv, pcie_chunk_elems, datatype, pcie_stream);
                domain->timer_pcie().Stop(pcie_stream);
                pcie_ok = (pcie_result == BackendResult::Success);
            }
        } else {
            domain->timer_fast().Start(stream);
            BackendResult result = FastBackendImpl::AllGather(
                sendbuff, recvbuff, sendcount, datatype, comm, stream);
            domain->timer_fast().Stop(stream);
            fast_ok = (result == BackendResult::Success);
        }

        DomainManager::GetInstance().RegisterStreamPending(
            stream, domain, op_key, plan, fast_ok, pcie_ok);

        return fast_ok ? BackendResult::Success : BackendResult::UnhandledError;
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
