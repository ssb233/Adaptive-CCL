#ifndef AMPCCL_CORE_DOMAIN_MANAGER_H_
#define AMPCCL_CORE_DOMAIN_MANAGER_H_

#include "domain.h"
#include "controller/algo_factory.h"
#include "common/log.h"
#include <unordered_map>
#include <mutex>
#include <memory>

namespace ampccl {

// Manages the global "dividing param" table keyed by *our* Comm (CommDomainKey),
// and the mapping from raw backend communicator to our Comm.
//
// - Table: our Comm (topology + ranks + size) -> CommDomain (controller + param cache).
//   Shared across ranks; reused when a communicator is destroyed and re-created
//   with the same topology/ranks/size.
// - Raw mapping: raw comm pointer -> our Comm key. Used only to find which
//   domain to use for a given collective call. Unregistering a raw comm does
//   not remove the domain.
class DomainManager {
public:
    static DomainManager& GetInstance() {
        static DomainManager instance;
        return instance;
    }

    // Get or create domain by *our* Comm key (used by CommInit after building key).
    CommDomain* GetOrCreateDomainByKey(const CommDomainKey& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        return GetOrCreateDomainByKeyLocked(key);
    }

    // Register raw communicator to our Comm. Call after backend CommInit.
    void RegisterRawComm(void* raw_comm, const CommDomainKey& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        (void)GetOrCreateDomainByKeyLocked(key);
        raw_to_key_[raw_comm] = key;
    }

    // Get domain for a raw communicator (for collective hooks).
    CommDomain* GetDomainByRawComm(void* raw_comm) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = raw_to_key_.find(raw_comm);
        if (it == raw_to_key_.end()) {
            return nullptr;
        }
        auto jt = key_to_domain_.find(it->second);
        if (jt == key_to_domain_.end()) {
            return nullptr;
        }
        return jt->second.get();
    }

    // Unregister raw communicator on CommDestroy. Does not remove the domain.
    void UnregisterRawComm(void* raw_comm) {
        std::lock_guard<std::mutex> lock(mutex_);
        raw_to_key_.erase(raw_comm);
    }

    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        raw_to_key_.clear();
        key_to_domain_.clear();
    }

private:
    DomainManager() = default;
    ~DomainManager() = default;
    DomainManager(const DomainManager&) = delete;
    DomainManager& operator=(const DomainManager&) = delete;

    CommDomain* GetOrCreateDomainByKeyLocked(const CommDomainKey& key) {
        auto it = key_to_domain_.find(key);
        if (it != key_to_domain_.end()) {
            AMPCCL_LOG(INFO, "Comm reused (existing): world_size=%d topology_hash=%llu",
                       key.world_size, static_cast<unsigned long long>(key.topology_hash));
            return it->second.get();
        }
        AMPCCL_LOG(INFO, "Comm created (new): world_size=%d topology_hash=%llu",
                   key.world_size, static_cast<unsigned long long>(key.topology_hash));
        CommDomain temp_domain(key, nullptr);
        auto algo = AlgoFactory::Create(temp_domain);
        auto controller = std::make_unique<AdaptiveController>(std::move(algo));
        auto domain = std::make_unique<CommDomain>(key, std::move(controller));
        CommDomain* ptr = domain.get();
        key_to_domain_[key] = std::move(domain);
        return ptr;
    }

    mutable std::mutex mutex_;
    std::unordered_map<CommDomainKey, std::unique_ptr<CommDomain>> key_to_domain_;
    std::unordered_map<void*, CommDomainKey> raw_to_key_;
};

}  // namespace ampccl

#endif  // AMPCCL_CORE_DOMAIN_MANAGER_H_
