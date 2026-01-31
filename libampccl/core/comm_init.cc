#include "comm_init.h"

#ifdef AMPCCL_ENABLE_PCIE
#include "comm.hpp"
#endif

namespace ampccl {

void InitPCIeForDomain(CommDomain* domain, int rank, int nranks) {
    if (!domain || nranks <= 0 || rank < 0 || rank >= nranks) {
        return;
    }
#ifdef AMPCCL_ENABLE_PCIE
    pcclComm_t pcie_comm = nullptr;
    pcclResult_t ret = pcclInit(rank, nranks, &pcie_comm);
    if (ret != pcclSuccess || !pcie_comm) {
        return;
    }
    domain->set_pcie_comm(pcie_comm);
    domain->set_pcie_rank(rank);
    domain->set_pcie_nranks(nranks);
#else
    (void)rank;
    (void)nranks;
#endif
}

}  // namespace ampccl
