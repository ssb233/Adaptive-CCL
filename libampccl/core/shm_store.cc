#include "shm_store.h"
#include "common/log.h"
#include <cstring>
#include <sstream>
#include <functional>
#include <algorithm>

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ampccl {

namespace {

#if defined(__linux__) || defined(__APPLE__)
constexpr bool kShmAvailable = true;
#else
constexpr bool kShmAvailable = false;
#endif

}  // namespace

std::string ShmParamStore::ShmNameForKey(const CommDomainKey& key) {
    size_t h = std::hash<CommDomainKey>{}(key);
    std::ostringstream os;
    os << "/ampccl_" << std::hex << h;
    return os.str();
}

size_t ShmParamStore::ShmSize() const {
    size_t header = sizeof(Header);
    size_t stat_slots = static_cast<size_t>(kMaxRanks) * sizeof(StatSlot);
    size_t param_header = sizeof(uint64_t) + sizeof(uint32_t);  // version + num_entries
    size_t param_entries = static_cast<size_t>(kMaxParamEntries) * sizeof(ParamEntry);
    return header + stat_slots + param_header + param_entries;
}

bool ShmParamStore::Attach(const CommDomainKey& key, int my_rank, int nranks) {
    if (!kShmAvailable || nranks <= 0 || my_rank < 0 || my_rank >= nranks) {
        return false;
    }
    if (nranks > kMaxRanks) {
        AMPCCL_LOG(WARN, "ShmStore: nranks %d > kMaxRanks %d, shm disabled", nranks, kMaxRanks);
        return false;
    }
    std::string name = ShmNameForKey(key);
    shm_size_ = ShmSize();
    my_rank_ = my_rank;
    nranks_ = nranks;

#if defined(__linux__) || defined(__APPLE__)
    int flags = O_RDWR;
    mode_t mode = 0666;
    shm_fd_ = shm_open(name.c_str(), O_CREAT | O_RDWR, mode);
    if (shm_fd_ < 0) {
        AMPCCL_LOG(WARN, "ShmStore: shm_open %s failed", name.c_str());
        return false;
    }
    struct stat st;
    if (fstat(shm_fd_, &st) != 0) {
        if (ftruncate(shm_fd_, static_cast<off_t>(shm_size_)) != 0) {
            AMPCCL_LOG(WARN, "ShmStore: ftruncate failed");
            close(shm_fd_);
            shm_fd_ = -1;
            return false;
        }
    } else if (st.st_size == 0) {
        if (ftruncate(shm_fd_, static_cast<off_t>(shm_size_)) != 0) {
            AMPCCL_LOG(WARN, "ShmStore: ftruncate failed");
            close(shm_fd_);
            shm_fd_ = -1;
            return false;
        }
    } else if (static_cast<size_t>(st.st_size) != shm_size_) {
        AMPCCL_LOG(WARN, "ShmStore: size mismatch");
        close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }
    base_ = mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (base_ == MAP_FAILED) {
        base_ = nullptr;
        close(shm_fd_);
        shm_fd_ = -1;
        AMPCCL_LOG(WARN, "ShmStore: mmap failed");
        return false;
    }
    Header* hdr = static_cast<Header*>(base_);
    if (hdr->magic != kMagic) {
        hdr->magic = kMagic;
        hdr->nranks = nranks;
        hdr->param_version = 0;
        hdr->pad = 0;
    }
    return true;
#else
    (void)key;
    return false;
#endif
}

ShmParamStore::~ShmParamStore() {
#if defined(__linux__) || defined(__APPLE__)
    if (base_ != nullptr && base_ != MAP_FAILED) {
        munmap(base_, shm_size_);
        base_ = nullptr;
    }
    if (shm_fd_ >= 0) {
        close(shm_fd_);
        shm_fd_ = -1;
    }
#endif
}

void ShmParamStore::WriteMyStat(int my_rank, const OpKey& op_key, const ExecStat& stat) {
    if (base_ == nullptr || my_rank < 0 || my_rank >= nranks_) {
        return;
    }
    char* p = static_cast<char*>(base_);
    p += sizeof(Header);
    StatSlot* slot = reinterpret_cast<StatSlot*>(p + my_rank * sizeof(StatSlot));
    slot->op = static_cast<int>(op_key.op);
    slot->bytes = static_cast<uint64_t>(op_key.bytes);
    slot->datatype = op_key.datatype;
    slot->fast_time = stat.fast_time;
    slot->pcie_time = stat.pcie_time;
    slot->fast_bytes = stat.fast_bytes;
    slot->pcie_bytes = stat.pcie_bytes;
    slot->fast_success = stat.fast_success ? 1 : 0;
    slot->pcie_success = stat.pcie_success ? 1 : 0;
    slot->valid = 1;
}

bool ShmParamStore::ReadAllStatsAndAggregate(ExecStat* out_global_stat, OpKey* out_op_key) const {
    if (base_ == nullptr || out_global_stat == nullptr || out_op_key == nullptr) {
        return false;
    }
    char* p = static_cast<char*>(base_);
    p += sizeof(Header);
    StatSlot* slots = reinterpret_cast<StatSlot*>(p);

    double max_fast_time = 0.0;
    double max_pcie_time = 0.0;
    uint64_t fast_bytes = 0;
    uint64_t pcie_bytes = 0;
    uint8_t fast_ok = 1;
    uint8_t pcie_ok = 1;
    bool any_valid = false;
    bool op_key_set = false;

    for (int r = 0; r < nranks_; ++r) {
        const StatSlot* s = &slots[r];
        if (s->valid == 0) {
            continue;
        }
        any_valid = true;
        if (!op_key_set) {
            out_op_key->op = static_cast<CollectiveType>(s->op);
            out_op_key->bytes = static_cast<size_t>(s->bytes);
            out_op_key->datatype = s->datatype;
            op_key_set = true;
        }
        if (s->fast_time > max_fast_time) {
            max_fast_time = s->fast_time;
        }
        if (s->pcie_time > max_pcie_time) {
            max_pcie_time = s->pcie_time;
        }
        fast_bytes = s->fast_bytes;
        pcie_bytes = s->pcie_bytes;
        if (s->fast_success == 0) {
            fast_ok = 0;
        }
        if (s->pcie_success == 0) {
            pcie_ok = 0;
        }
    }
    if (!any_valid) {
        return false;
    }
    out_global_stat->fast_time = max_fast_time;
    out_global_stat->pcie_time = max_pcie_time;
    out_global_stat->fast_bytes = static_cast<size_t>(fast_bytes);
    out_global_stat->pcie_bytes = static_cast<size_t>(pcie_bytes);
    out_global_stat->fast_success = (fast_ok != 0);
    out_global_stat->pcie_success = (pcie_ok != 0);
    return true;
}

void ShmParamStore::ReadParams(ParamCache* cache) const {
    if (base_ == nullptr || cache == nullptr) {
        return;
    }
    char* p = static_cast<char*>(base_);
    size_t off = sizeof(Header) + static_cast<size_t>(kMaxRanks) * sizeof(StatSlot);
    uint64_t* pver = reinterpret_cast<uint64_t*>(p + off);
    uint32_t* pnum = reinterpret_cast<uint32_t*>(p + off + sizeof(uint64_t));
    ParamEntry* entries = reinterpret_cast<ParamEntry*>(p + off + sizeof(uint64_t) + sizeof(uint32_t));

    uint32_t n = *pnum;
    if (n > static_cast<uint32_t>(kMaxParamEntries)) {
        n = kMaxParamEntries;
    }
    std::vector<std::pair<OpKey, ParamValue>> snapshot;
    snapshot.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        const ParamEntry& e = entries[i];
        OpKey key;
        key.op = static_cast<CollectiveType>(e.op);
        key.bytes = static_cast<size_t>(e.bytes);
        key.datatype = e.datatype;
        ParamValue val(e.alpha, e.use_pcie != 0, e.fast_bw, e.pcie_bw);
        snapshot.emplace_back(key, val);
    }
    cache->Clear();
    cache->SetFrom(snapshot);
    (void)pver;
}

void ShmParamStore::WriteParams(const ParamCache& cache) {
    if (base_ == nullptr) {
        return;
    }
    std::vector<std::pair<OpKey, ParamValue>> snapshot;
    cache.GetAll(&snapshot);
    if (snapshot.size() > static_cast<size_t>(kMaxParamEntries)) {
        snapshot.resize(kMaxParamEntries);
    }
    char* p = static_cast<char*>(base_);
    size_t off = sizeof(Header) + static_cast<size_t>(kMaxRanks) * sizeof(StatSlot);
    uint64_t* pver = reinterpret_cast<uint64_t*>(p + off);
    uint32_t* pnum = reinterpret_cast<uint32_t*>(p + off + sizeof(uint64_t));
    ParamEntry* entries = reinterpret_cast<ParamEntry*>(p + off + sizeof(uint64_t) + sizeof(uint32_t));

    *pnum = static_cast<uint32_t>(snapshot.size());
    for (size_t i = 0; i < snapshot.size(); ++i) {
        const auto& kv = snapshot[i];
        ParamEntry& e = entries[i];
        e.op = static_cast<int>(kv.first.op);
        e.bytes = static_cast<uint64_t>(kv.first.bytes);
        e.datatype = kv.first.datatype;
        e.alpha = kv.second.alpha;
        e.use_pcie = kv.second.use_pcie ? 1 : 0;
        e.pad[0] = e.pad[1] = e.pad[2] = e.pad[3] = 0;
        e.fast_bw = kv.second.fast_bw;
        e.pcie_bw = kv.second.pcie_bw;
    }
    (*pver)++;
}

}  // namespace ampccl
