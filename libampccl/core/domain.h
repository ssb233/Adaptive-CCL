#ifndef AMPCCL_CORE_DOMAIN_H_
#define AMPCCL_CORE_DOMAIN_H_

#include "cache/param_cache.h"
#include "controller/controller.h"
#include <vector>
#include <cstdint>
#include <memory>
#include <functional>

namespace ampccl {

struct CommDomainKey {
    int world_size;
    std::vector<int> ranks;
    uint64_t topology_hash;

    bool operator==(const CommDomainKey& other) const {
        if (world_size != other.world_size ||
            topology_hash != other.topology_hash ||
            ranks.size() != other.ranks.size()) {
            return false;
        }
        for (size_t i = 0; i < ranks.size(); ++i) {
            if (ranks[i] != other.ranks[i]) {
                return false;
            }
        }
        return true;
    }
};

// Hash function for CommDomainKey
namespace std {
template <>
struct hash<CommDomainKey> {
    size_t operator()(const CommDomainKey& key) const {
        size_t h = std::hash<int>{}(key.world_size);
        h ^= std::hash<uint64_t>{}(key.topology_hash) << 1;
        for (int rank : key.ranks) {
            h ^= std::hash<int>{}(rank) << 1;
        }
        return h;
    }
};
}  // namespace std

class CommDomain {
public:
    CommDomainKey key;
    std::unique_ptr<AdaptiveController> controller;
    ParamCache param_cache;

    // PCIe (PCCL) communicator state - set in CommInit via InitPCIeForDomain
    void* pcie_comm() const { return pcie_comm_; }
    void set_pcie_comm(void* c) { pcie_comm_ = c; }
    int pcie_rank() const { return pcie_rank_; }
    void set_pcie_rank(int r) { pcie_rank_ = r; }
    int pcie_nranks() const { return pcie_nranks_; }
    void set_pcie_nranks(int n) { pcie_nranks_ = n; }

    CommDomain(const CommDomainKey& k, std::unique_ptr<AdaptiveController> ctrl)
        : key(k), controller(std::move(ctrl)),
          pcie_comm_(nullptr), pcie_rank_(-1), pcie_nranks_(0) {}

private:
    void* pcie_comm_;   // pcclComm_t (opaque)
    int pcie_rank_;
    int pcie_nranks_;
};

}  // namespace ampccl

#endif  // AMPCCL_CORE_DOMAIN_H_
