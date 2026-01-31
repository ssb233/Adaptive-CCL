#include "stream_sync.h"
#include "domain_manager.h"
#include "domain.h"
#include "telemetry/stats.h"
#include "common/log.h"
#include <optional>

#ifdef AMPCCL_ENABLE_PCIE
#include "comm.hpp"
#endif

namespace ampccl {

void OnStreamSynchronized(void* stream) {
    std::optional<PendingCollective> pending =
        DomainManager::GetInstance().TakeStreamPending(stream);
    if (!pending) {
        return;
    }
    CommDomain* domain = pending->domain;
    if (!domain || !domain->controller) {
        return;
    }

#ifdef AMPCCL_ENABLE_PCIE
    if (domain->pcie_comm() && domain->pcie_stream() && pending->plan.use_pcie) {
        pcclResult_t ret = pcclSynchronizeStream(
            static_cast<pcclComm_t>(domain->pcie_comm()),
            static_cast<pcclStream_t>(domain->pcie_stream()));
        (void)ret;
    }
#endif

    domain->timer_fast().Synchronize();
    if (pending->plan.use_pcie) {
        domain->timer_pcie().Synchronize();
    }

    ExecStat stat;
    stat.fast_time = domain->timer_fast().ElapsedSeconds();
    stat.pcie_time = pending->plan.use_pcie ? domain->timer_pcie().ElapsedSeconds() : 0.0;
    stat.fast_bytes = pending->plan.fast_bytes;
    stat.pcie_bytes = pending->plan.pcie_bytes;
    stat.fast_success = pending->fast_success;
    stat.pcie_success = pending->pcie_success;

    domain->EnsureShmAttached();
    int nranks = domain->pcie_nranks();
    ShmParamStore* shm = domain->shm_store();
    if (nranks > 1 && shm->IsAttached()) {
        shm->WriteMyStat(domain->pcie_rank(), pending->op_key, stat);
        AMPCCL_LOG(INFO, "StreamSync: wrote stat to shm (rank %d) op_key.bytes=%zu fast_time=%.6fs pcie_time=%.6fs",
                   domain->pcie_rank(), pending->op_key.bytes, stat.fast_time, stat.pcie_time);
    } else {
        domain->controller->Update(pending->op_key, stat, domain->param_cache);
        AMPCCL_LOG(INFO, "StreamSync: op_key.bytes=%zu fast_time=%.6fs pcie_time=%.6fs fast_bytes=%zu pcie_bytes=%zu",
                   pending->op_key.bytes, stat.fast_time, stat.pcie_time,
                   stat.fast_bytes, stat.pcie_bytes);
    }
}

}  // namespace ampccl
