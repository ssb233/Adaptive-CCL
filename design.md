# Adaptive Multi-Path Collective Communication Layer (AMP-CCL)

> A transparent LD_PRELOAD-based adaptive multi-path collective layer for NCCL / HCCL + PCIe.

------

# 1. Design Goals

AMP-CCL aims to:

1. Transparently intercept NCCL/HCCL collectives via LD_PRELOAD
2. Split one collective into two parallel sub-collectives
   - **Fast backend** (NVLink / IB / HCCS via NCCL/HCCL)
   - **PCIe backend** (vendor provided PCIe CCL)
3. Dynamically adapt split ratio via feedback control
4. Be backend-agnostic via template-based abstraction
5. Be domain-aware (bound to communicator / topology)
6. Support multiple adaptive algorithms via factory pattern
7. Support drop-back to fast-only when PCIe is inefficient

------

# 2. High Level Architecture

```
User Application
        |
   NCCL / HCCL API
        |
   (LD_PRELOAD)
        |
   AMP-CCL
   ┌─────────────────────────────┐
   │  Virtual Collective Layer   │
   │  + Adaptive Controller      │
   │  + Planner                  │
   │  + Param Cache              │
   └──────────┬──────────┬───────┘
              |          |
      Fast Backend   PCIe Backend
   (NCCL/HCCL)     (pcieccl / vendor)
```

------

# 3. File & Module Layout

```
libampccl/
│
├── hook/
│   ├── hccl_hook.cc
│   ├── nccl_hook.cc
│
├── core/
│   ├── virtual_collective.h
│   ├── domain.h
│   ├── domain_manager.h
│   ├── planner.h
│
├── controller/
│   ├── controller.h
│   ├── algo_base.h
│   ├── algo_factory.h
│   ├── algo_tcp.h
│   ├── algo_dcqcn.h
│
├── backend/
│   ├── backend_base.h
│   ├── fast_backend.h
│   ├── pcie_backend.h
│
├── cache/
│   └── param_cache.h
│
├── telemetry/
│   ├── timer.h
│   └── stats.h
│
└── common/
    ├── op_key.h
    ├── config.h
```

------

# 4. Communication Domain

Adaptive behavior must be bound to a **communication domain** (topology + communicator).

```
struct CommDomainKey {
    int world_size;
    std::vector<int> ranks;
    uint64_t topology_hash;
};
class CommDomain {
public:
    CommDomainKey key;
    std::unique_ptr<AdaptiveController> controller;
    ParamCache param_cache;
};
```

All collectives operate within a `CommDomain`.

`DomainManager` maps NCCL/HCCL communicator → CommDomain.

------

# 5. Param Cache (Learning Table)

Each domain learns optimal parameters per operator.

```
struct OpKey {
    enum Type { AllReduce, AllGather, ReduceScatter };
    Type op;
    size_t bytes;
    int datatype;
};

struct ParamValue {
    double alpha;      // fast backend ratio
    bool use_pcie;
    double fast_bw;
    double pcie_bw;
};
class ParamCache {
    std::unordered_map<OpKey, ParamValue> table;
};
```

This is what gives you **per-op, per-size adaptive memory**.

------

# 6. Hook Layer (LD_PRELOAD)

Exports symbols identical to NCCL/HCCL:

```
hcclResult_t hcclAllReduce(...)
ncclResult_t ncclAllReduce(...)
```

Hook logic:

```
1. Find CommDomain
2. Call VirtualCollective::AllReduce(domain, args...)
```

------

# 7. Virtual Collective

This is the heart of the system.

```
class VirtualCollective {
public:
    static Result AllReduce(CommDomain*, Args...);
};
```

Execution pipeline:

```
1. Build OpKey
2. ParamCache lookup
3. Controller suggests alpha
4. Planner builds split plan
5. Launch fast + PCIe backend
6. Wait
7. Measure times
8. Controller.update()
9. Cache.update()
```

------

# 8. Planner (Split & Drop-back Logic)

```
struct Plan {
    size_t fast_bytes;
    size_t pcie_bytes;
    bool use_pcie;
};
```

Responsibilities:

- Enforce minimum chunk size
- Disable PCIe for small messages
- Clamp alpha
- Handle alignment

------

# 9. Backend Abstraction (Template)

```
template<typename Backend>
class CollectiveBackend {
public:
    static Result AllReduce(...);
};
```

Backends:

```
struct HCCLBackend {};
struct NCCLBackend {};
struct PCIeBackend {};
```

This gives you **compile-time polymorphism** and zero-overhead abstraction.

------

# 10. Telemetry

Each execution produces:

```
struct ExecStat {
    double fast_time;
    double pcie_time;
    size_t fast_bytes;
    size_t pcie_bytes;
};
```

Measured via CUDA events or CPU timers.

------

# 11. Adaptive Algorithm (Factory Pattern)

```
class AdaptiveAlgo {
public:
    virtual double suggest(const ParamValue&) = 0;
    virtual void update(const ExecStat&) = 0;
};
class AlgoFactory {
public:
    static std::unique_ptr<AdaptiveAlgo> create(const CommDomain&);
};
```

Algorithms:

- TCP-style AIMD
- DCQCN-style
- Future plugins

------

# 12. Drop-back Mechanism

Handled by:

- Algorithm (via alpha → 0)
- Planner (via `use_pcie = false`)

Conditions:

- PCIe BW too low
- Message too small
- PCIe unstable

------

# 13. Why This Architecture Works

This system is:

| Requirement         | Supported      |
| ------------------- | -------------- |
| LD_PRELOAD          | Yes            |
| Vendor independent  | Yes            |
| Multiple algorithms | Factory        |
| Topology aware      | CommDomain     |
| Per-op learning     | ParamCache     |
| Template backends   | Yes            |
| Drop-back           | Planner + Algo |

This is essentially a **software-defined collective transport layer**.