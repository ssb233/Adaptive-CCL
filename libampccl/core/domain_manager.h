#ifndef AMPCCL_CORE_DOMAIN_MANAGER_H_
#define AMPCCL_CORE_DOMAIN_MANAGER_H_

#include "domain.h"
#include "controller/algo_factory.h"
#include <unordered_map>
#include <mutex>
#include <memory>

namespace ampccl {

// Maps NCCL/HCCL communicator handles to CommDomain
class DomainManager {
public:
    static DomainManager& GetInstance() {
        static DomainManager instance;
        return instance;
    }

    // Get or create domain for a communicator
    CommDomain* GetOrCreateDomain(void* comm, const CommDomainKey& key) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = comm_to_domain_.find(comm);
        if (it != comm_to_domain_.end()) {
            return it->second.get();
        }

        // Create new domain
        // Create a temporary domain for factory (needs proper initialization)
        CommDomain temp_domain(key, nullptr);
        auto algo = AlgoFactory::Create(temp_domain);
        auto controller = std::make_unique<AdaptiveController>(std::move(algo));
        auto domain = std::make_unique<CommDomain>(key, std::move(controller));

        CommDomain* domain_ptr = domain.get();
        comm_to_domain_[comm] = std::move(domain);

        return domain_ptr;
    }

    // Get domain for a communicator (returns nullptr if not found)
    CommDomain* GetDomain(void* comm) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = comm_to_domain_.find(comm);
        if (it != comm_to_domain_.end()) {
            return it->second.get();
        }
        return nullptr;
    }

    // Remove domain (when communicator is destroyed)
    void RemoveDomain(void* comm) {
        std::lock_guard<std::mutex> lock(mutex_);
        comm_to_domain_.erase(comm);
    }

    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        comm_to_domain_.clear();
    }

private:
    DomainManager() = default;
    ~DomainManager() = default;
    DomainManager(const DomainManager&) = delete;
    DomainManager& operator=(const DomainManager&) = delete;

    mutable std::mutex mutex_;
    std::unordered_map<void*, std::unique_ptr<CommDomain>> comm_to_domain_;
};

}  // namespace ampccl

#endif  // AMPCCL_CORE_DOMAIN_MANAGER_H_
