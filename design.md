# Adaptive-CCL 设计说明

本文档描述 Adaptive Multi-Path Collective Communication Layer（AMP-CCL）的完整架构、调用链，以及如何集成与调用 PCIeCCL（pcieccl / PCCL）。

---

## 1. 概述与目标

AMP-CCL 通过 **LD_PRELOAD** 透明拦截 NCCL / HCCL 的集合通信接口，将一次集合通信拆成两条并行路径：

- **快路径（Fast Backend）**：走 NCCL / HCCL 原有通道（如 NVLink、IB、HCCS 等）。
- **PCIe 路径（PCIe Backend）**：走 PCIe 的集合通信库（当前实现为 **PCIeCCL / PCCL**）。

目标包括：

- 对应用透明，无需改代码，仅通过 LD_PRELOAD 注入。
- 按通信域（拓扑 + 通信子）做**自适应分片**，用反馈控制动态调整快路径与 PCIe 路径的字节比例（alpha）。
- 多 Rank 时**参数表全局一致**：所有 Rank 看到同一份参数，仅 Rank 0 根据整体耗时更新参数（通过共享内存）。
- 支持在 PCIe 效率低时回退到仅快路径。

---

## 2. 整体架构

```
用户应用
    |
NCCL / HCCL API（集合通信、流同步等）
    |
LD_PRELOAD 注入
    |
AMP-CCL
├── Hook 层（nccl_hook / hccl_hook）
│   └── 拦截 CommInit、CommDestroy、AllReduce、AllGather、SynchronizeStream 等
├── 虚拟集合层（VirtualCollective）
│   ├── 参数来源：单 Rank 用本地 ParamCache；多 Rank 用共享内存 ShmParamStore
│   ├── 规划器（Planner）生成分片计划（fast_bytes / pcie_bytes）
│   ├── 快路径：FastBackend → 原始 NCCL/HCCL
│   └── PCIe 路径：PCIeBackend → PCIeCCL（pcclInit、pcclSubmit、pcclStream 等）
├── 流同步（OnStreamSynchronized）
│   └── 取 PendingCollective、同步计时、写本 Rank 统计到共享内存或本地 Update
└── 通信域管理（DomainManager + CommDomain）
    └── 每个进程一个 CommDomain，key 相同；维护 ParamCache、PCIe 资源、计时器、ShmParamStore
```

- **实际通信子（comm）的创建与销毁**仍由 NCCL / HCCL 负责；我们只通过 **raw_comm → CommDomainKey** 找到对应的 **CommDomain**，在域内维护参数表与 PCIe 资源（pcie_comm、pcie_stream 等）。

---

## 3. 模块与目录结构

```
libampccl/
├── hook/
│   ├── nccl_hook.cc      # 拦截 NCCL + cudaStreamSynchronize
│   └── hccl_hook.cc      # 拦截 HCCL + aclrtSynchronizeStream
├── core/
│   ├── domain_key.h      # CommDomainKey 及 hash（供 shm 命名等）
│   ├── domain.h          # CommDomain（key、param_cache、controller、PCIe 状态、计时器、ShmParamStore）
│   ├── domain_manager.h  # DomainManager：key↔Domain、raw_comm↔key、stream↔PendingCollective
│   ├── comm_init.h/cc    # BuildKeyFromNccl/HcclInit、InitPCIeForDomain（pcclInit、pcclCreateStream）
│   ├── virtual_collective.h  # VirtualCollective：AllReduce/AllGather 入口与分片执行
│   ├── planner.h         # Planner：按 alpha、use_pcie 生成 Plan（fast_bytes、pcie_bytes）
│   ├── stream_sync.h/cc # OnStreamSynchronized：消费 Pending、同步计时、写统计或 Update
│   └── shm_store.h/cc    # ShmParamStore：共享内存布局、Attach、WriteMyStat、ReadParams、WriteParams
├── controller/
│   ├── controller.h      # AdaptiveController（SuggestAlpha、Update）
│   ├── algo_base.h       # AdaptiveAlgo 接口
│   ├── algo_factory.h    # 按配置选择算法（TCP/DCQCN/STATIC）
│   ├── algo_tcp.h
│   └── algo_dcqcn.h
├── backend/
│   ├── backend_base.h    # BackendResult、模板 BackendBase
│   ├── fast_backend.h/cc # FastBackendImpl：直接调 NCCL/HCCL
│   └── pcie_backend.h/cc # PCIeBackendImpl：调 PCIeCCL（CommDomain 提供 pcie_comm、pcie_stream）
├── cache/
│   └── param_cache.h     # ParamCache（OpKey → ParamValue）、GetAll/SetFrom 供 shm 序列化
├── telemetry/
│   ├── timer.h           # Timer（CUDA/ACL 事件或 CPU 计时），Start(stream)、Stop(stream)、Synchronize()
│   └── stats.h           # ExecStat（fast_time、pcie_time、bytes、success）
├── common/
│   ├── op_key.h          # OpKey（op、bytes、datatype）、CollectiveType
│   ├── config.h          # AMPCCL_ENABLE、AMPCCL_ALGO、AMPCCL_MIN_*、AMPCCL_ENABLE_PCIE 等
│   └── log.h             # 日志级别与 AMPCCL_LOG
```

---

## 4. 通信域（CommDomainKey、CommDomain、DomainManager）

### 4.1 通信域的身份：CommDomainKey

集合通信的“逻辑通信域”由 **CommDomainKey** 标识，与具体进程无关，同一拓扑下各 Rank 的 key 一致：

- `world_size`：秩数。
- `ranks`：秩列表（如 0..n-1）。
- `topology_hash`：由 NCCL/HCCL 的 commId 等推导出的拓扑哈希。

**我们不管理 NCCL/HCCL 的 comm 列表**，只关心“有哪些 Rank”；comm 的创建与销毁由 NCCL/HCCL 完成。

### 4.2 CommDomain（每进程一个）

每个进程内，对应当前使用的通信子，会有一个 **CommDomain** 实例（通过 DomainManager 按 key 获取或创建）。其内部只维护**本进程需要**的内容：

- **key**：同上，标识逻辑域。
- **param_cache**：参数表（单 Rank 时本地读写；多 Rank 时由共享内存提供，见下）。
- **controller**：自适应算法（SuggestAlpha、Update），多 Rank 时仅 Rank 0 用其写回参数。
- **PCIe 相关**（由 InitPCIeForDomain 在 CommInit 后设置）：
  - `pcie_comm`：PCCL 的 `pcclComm_t`（来自 `pcclInit`）。
  - `pcie_rank`、`pcie_nranks`：本进程 Rank 与总秩数。
  - `pcie_stream`：PCCL 的 `pcclStream_t`（来自 `pcclCreateStream`），PCIe 路径专用，放在 domain 内统一管理。
- **计时器**：`timer_fast`、`timer_pcie`，分别挂在用户 stream 和 `pcie_stream` 上，用于在 SynchronizeStream 时得到 fast_time / pcie_time。
- **ShmParamStore**：多 Rank 时按需 attach 的共享内存，用于“每 Rank 写本 Rank 统计、Rank 0 聚合并写回参数表”。

### 4.3 DomainManager

- **key → CommDomain**：同一 key 在单进程内只对应一个 CommDomain，comm 销毁后 domain 不删，便于复用参数。
- **raw_comm → CommDomainKey**：根据 NCCL/HCCL 的 comm 句柄找到 key，再找到 CommDomain。
- **stream → PendingCollective**：某 stream 上若刚执行过我们发起的集合通信，在 SynchronizeStream 时用该 pending 做计时与统计写入（见第 7 节）。

CommInit 被拦截后：用 (nranks, commId, rank) 构建 CommDomainKey，GetOrCreateDomainByKey，RegisterRawComm(raw_comm, key)，并调用 InitPCIeForDomain(domain, rank, nranks)。  
CommDestroy 被拦截后：仅 UnregisterRawComm(raw_comm)，不删除 Domain。

---

## 5. 多 Rank 与共享内存（ShmParamStore）

为保证**所有 Rank 看到同一份参数表**，且**只用整体集合通信时间（如各 Rank 的 max）来调参**，采用共享内存方案：

- **ShmParamStore** 按 CommDomainKey 的 hash 命名（如 `/ampccl_<hex>`），同一 key 的进程 attach 到同一块共享段。
- **布局**：Header（magic、nranks、param_version）+ 每 Rank 一个 **StatSlot**（op、bytes、datatype、fast_time、pcie_time、fast_bytes、pcie_bytes、success、valid）+ **参数区**（version、num_entries、ParamEntry[]）。
- **单 Rank（nranks==1）**：不 attach shm，ParamCache 与 Controller 的 Update 均在本地完成。
- **多 Rank（nranks>1）**：
  - **SynchronizeStream 时**：每个 Rank 只把本 Rank 的 ExecStat 写入自己的 StatSlot（WriteMyStat），**不**在本进程调用 controller->Update。
  - **下一次集合通信入口**（如 AllReduce/AllGather 被调用时）：  
    - **Rank 0**：从 shm 中 ReadAllStatsAndAggregate（对各 Rank 的 fast_time、pcie_time 取 max 等），得到全局 ExecStat，再 controller->Update(agg_op_key, global_stat, param_cache)，最后 WriteParams(domain->param_cache) 写回 shm。  
    - **所有 Rank**：ReadParams(&domain->param_cache)，用 shm 中的参数表**整体覆盖**本地 cache（以 shm 为唯一真相）。
- 这样：测量的是**整体**时间（max over ranks），8 个 Rank 看到的参数表一致，且**只有 Rank 0 修改**参数表。

---

## 6. Hook 层与调用链

### 6.1 拦截的符号（HCCL / NCCL）

- **HCCL**：`HcclGetUniqueId`、`HcclCommInitRank`、`HcclCommDestroy`、`HcclAllReduce`、`HcclAllGather`、`HcclReduceScatter`、`HcclBroadcast`，以及 **aclrtSynchronizeStream**（用于在流同步时做计时与统计写 shm）。
- **NCCL**：对应 nccl 符号，以及 **cudaStreamSynchronize**。

原始实现通过 **dlopen/dlsym** 从 libhccl.so / libnccl.so、libascendcl.so 或 libacl.so、libcudart.so 等获取，未开启自适应（如 `AMPCCL_ENABLE!=1`）时直接转调原始接口。

### 6.2 调用链概览

1. **CommInit**  
   → 调原始 CommInit → 用 (nranks, commId, rank) 建 CommDomainKey → RegisterRawComm → InitPCIeForDomain（pcclInit、pcclCreateStream，并设置 domain 的 pcie_comm、pcie_rank、pcie_nranks、pcie_stream）。

2. **AllReduce / AllGather**  
   → 根据 raw_comm 取 CommDomain → EnsureShmAttached（多 Rank 时）→ 若 Rank 0 则从 shm 聚合并 Update 再 WriteParams → 所有 Rank ReadParams 刷新 param_cache → ParamCache Lookup、Controller SuggestAlpha、Planner CreatePlan → 在**用户 stream** 上录 timer_fast、发快路径、再录 timer_fast；在 **pcie_stream** 上录 timer_pcie、发 PCIe 路径、再录 timer_pcie → RegisterStreamPending(stream, domain, op_key, plan, fast_ok, pcie_ok)。

3. **SynchronizeStream（aclrtSynchronizeStream / cudaStreamSynchronize）**  
   → 先调原始 SynchronizeStream → OnStreamSynchronized(stream)：TakeStreamPending(stream)，若有 pending 则同步 PCIe stream（若需要）、timer_fast/timer_pcie 的 Synchronize，得到 ExecStat；多 Rank 且 shm 已 attach 则 WriteMyStat，否则本地 controller->Update。

4. **CommDestroy**  
   → UnregisterRawComm，不删 Domain。

---

## 7. 虚拟集合层与流同步、计时

### 7.1 VirtualCollective 执行流程（以 AllReduce 为例）

1. 根据 count、datatype 构造 **OpKey**（op、bytes、datatype）。
2. **多 Rank**：EnsureShmAttached；Rank 0 从 shm ReadAllStatsAndAggregate → Update → WriteParams；所有 Rank ReadParams 覆盖 param_cache。
3. ParamCache **Lookup(op_key)**，Controller **SuggestAlpha**，Planner **CreatePlan(op_key.bytes, alpha, use_pcie)** 得到 Plan（fast_bytes、pcie_bytes、use_pcie）。
4. **发快路径**：在**用户 stream** 上 `timer_fast.Start(stream)` → FastBackendImpl::AllReduce(..., stream) → `timer_fast.Stop(stream)`。
5. **发 PCIe 路径**（若 plan.use_pcie 且 plan.pcie_bytes>0）：在 **domain->pcie_stream()** 上 `timer_pcie.Start(pcie_stream)` → PCIeBackendImpl::AllReduce(domain, ..., pcie_stream) → `timer_pcie.Stop(pcie_stream)`。  
   - 不在 collective 内做任何 sync，保证透明性。
6. **RegisterStreamPending**(stream, domain, op_key, plan, fast_ok, pcie_ok)，然后返回。

### 7.2 OnStreamSynchronized（流同步时）

1. **TakeStreamPending(stream)**，若无 pending 则直接返回。
2. 若启用了 PCIe 且本次用了 PCIe，则调用 **pcclSynchronizeStream(domain->pcie_comm(), domain->pcie_stream())**，保证 PCIe 任务完成。
3. **timer_fast.Synchronize()**、**timer_pcie.Synchronize()**（若用了 PCIe），然后从 timer 取 **ElapsedSeconds()** 得到 fast_time、pcie_time，与 plan 中的 bytes、pending 中的 success 拼成 **ExecStat**。
4. **多 Rank 且 shm 已 attach**：**WriteMyStat**(my_rank, op_key, stat)，不调用 controller->Update。  
   **单 Rank 或 shm 未 attach**：**domain->controller->Update**(op_key, stat, domain->param_cache)。

这样，**同步发生在用户调用的 SynchronizeStream 处**，计时与参数更新都对齐到该点，对用户透明。

---

## 8. PCIe 后端与 PCIeCCL 的调用方式

### 8.1 依赖与编译

- 启用 PCIe 路径时需定义 **AMPCCL_ENABLE_PCIE**，并链接 **pcieccl** 提供的头与库（如 `comm.hpp`、`ir.hpp`、`libpccl.so`）。
- 在 **InitPCIeForDomain**（comm_init.cc）中：  
  - 调用 **pcclInit(rank, nranks, &pcie_comm)**，得到 `pcclComm_t`，存入 domain->set_pcie_comm(...)。  
  - 调用 **pcclCreateStream(pcie_comm, &pcie_stream)**，得到 `pcclStream_t`，存入 domain->set_pcie_stream(...)。  
  即：每个 CommDomain 拥有自己的 **pcie_comm** 和 **pcie_stream**，由 AMP-CCL 在 CommInit 时创建，comm 生命周期内复用。

### 8.2 PCIeBackend 如何调 PCIeCCL

- **入口**：VirtualCollective 在发 PCIe 路径时调用  
  `PCIeBackendImpl::AllReduce(domain, pcie_send, pcie_recv, count, datatype, op, pcie_stream)`  
  其中 `pcie_stream = domain->pcie_stream()`，发送/接收指针为按 plan 切分后的 buffer 偏移。
- **AllReduce（2 秩）**（pcie_backend.cc）：  
  - 用 domain 的 **pcie_comm**、**pcie_rank**、**pcie_stream**。  
  - 按 rank 构造 **IRProgram**（D2H、H2H_REDUCE、H2D 等指令）。  
  - 调用 **pcclSubmit(comm, program, sendbuff, recvbuff, count, pcie_stream)**，**不**在 Backend 内调用 pcclSynchronizeStream；同步留给上层 **OnStreamSynchronized** 中统一 **pcclSynchronizeStream(domain->pcie_comm(), domain->pcie_stream())**。
- **AllGather（2 秩）**：同样用 domain 的 pcie_comm / pcie_stream，构造 2 秩 AllGather 的 IRProgram，**pcclSubmit(..., pcie_stream)**，不在后端内部 sync。
- **ReduceScatter / Broadcast**：当前为桩实现，直接返回 Success。

### 8.3 小结

- **Comm 与 Stream**：由 AMP-CCL 在 **InitPCIeForDomain** 里调用 **pcclInit**、**pcclCreateStream** 得到，并挂在 **CommDomain** 上。  
- **执行**：PCIe 路径使用 **domain->pcie_stream()**，通过 **pcclSubmit** 提交；**pcclSynchronizeStream** 只在 **OnStreamSynchronized** 中调用，与用户侧的 aclrtSynchronizeStream / cudaStreamSynchronize 语义一致，保证透明性。

---

## 9. 规划器与控制器

- **Planner**：根据 total_bytes、alpha、use_pcie_hint 生成 **Plan**（fast_bytes、pcie_bytes、use_pcie）。会做最小消息/最小分块检查、对齐等；若不应走 PCIe（消息过小或 use_pcie 为 false），则 plan 仅快路径。
- **Controller / ParamCache**：ParamCache 存 (OpKey → ParamValue)（alpha、use_pcie、fast_bw、pcie_bw）。Controller 的 SuggestAlpha 用于本次分片，Update 用 ExecStat 更新算法内部状态并写回 ParamValue；多 Rank 时只有 Rank 0 执行 Update，并通过 ShmParamStore 写回共享内存。

---

## 10. 配置与日志

- **config.h**：如 AMPCCL_ENABLE、AMPCCL_ALGO、AMPCCL_MIN_MSG_SIZE、AMPCCL_MIN_CHUNK_SIZE、AMPCCL_ENABLE_PCIE 等，通过环境变量读取。
- **log.h**：AMPCCL_LOG(level, ...)，级别由环境变量或 SetLogLevel 控制，用于排查分片、计时、shm 与 PCIe 调用等。

---

## 11. 小结

| 项目         | 说明 |
|--------------|------|
| 透明性       | LD_PRELOAD 拦截 NCCL/HCCL 与流同步 API；同步与计时在用户调用的 SynchronizeStream 处完成。 |
| 通信域       | CommDomainKey 标识逻辑域；每进程一个 CommDomain，只维护本进程所需的参数表与 PCIe 资源；comm 由 NCCL/HCCL 管理。 |
| 多 Rank 一致  | 通过 ShmParamStore 共享“每 Rank 统计”与“唯一参数表”；Rank 0 聚合并写回，所有 Rank 读同一份参数。 |
| PCIeCCL 使用 | InitPCIeForDomain 中 pcclInit、pcclCreateStream；PCIeBackend 使用 domain->pcie_comm()、domain->pcie_stream()，仅 pcclSubmit；pcclSynchronizeStream 在 OnStreamSynchronized 中统一调用。 |

以上即为当前完整设计架构及 PCIeCCL 的集成与调用方式说明。
