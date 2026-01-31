# 多 Rank 下通信域与参数一致性设计

## 1. 当前问题

- **每进程一份状态**：每个 rank 是独立进程，各自有 `DomainManager`、`CommDomain`、`param_cache`、`controller`。虽然 `CommDomainKey`（world_size、ranks、topology_hash）在各 rank 上相同，但数据是**进程本地**的。
- **计时不一致**：各 rank 上 fast_time、pcie_time 可能不同（拓扑、负载、PCIe 路径等），若各自用本地计时去调参，每个进程会得到不同的 alpha/use_pcie。
- **参数表发散**：各 rank 各自维护 param_cache 并调用 controller->Update()，会导致 8 个进程上的参数表逐渐不一致，集合通信的“整体”行为无法由一份统一参数控制。
- **期望**：
  - 用**整体**集合通信时间（例如所有 rank 的 max）来驱动调参。
  - **所有 rank 看到同一份参数表**，且**只有 rank 0 能修改**（或集中更新后大家一致读取）。

## 2. 方案概览

| 方案 | 思路 | 优点 | 缺点 |
|------|------|------|------|
| **A. 共享内存** | 把 per-rank 计时和“唯一参数表”放进共享内存；rank 0 聚合、更新并写回，其他 rank 只读 | 不增加集合通信次数，逻辑清晰，单点写 | 需要约定共享内存布局与进程间同步 |
| **B. 用现有通信做归约/广播** | 在 SyncStream 或下一次 collective 时，用 AllReduce(max) 得到全局时间，rank 0 更新后把新参数广播给所有 rank | 不依赖共享内存 | 需在 hook 里多一次 collective 或复用现有 collective，实现与死锁风险需谨慎 |

下面只展开**方案 A（共享内存）**的推荐设计；方案 B 可作为后续优化（例如无共享内存环境）。

## 3. 方案 A：共享内存 + Rank 0 单点更新

### 3.1 语义约定

- **全局时间**：一次 collective 的“整体”时间取为所有 rank 上该次 collective 的 **max(fast_time), max(pcie_time)**（或你认为合理的其它聚合方式）。
- **参数表**：所有 rank 共用同一份 param_cache（以及 controller 内部状态，若需要）；**只有 rank 0 执行 controller->Update()** 并写回共享区；其他 rank 在每次 collective 前从共享区**只读**加载参数。

### 3.2 共享内存布局（示意）

- 用 **CommDomainKey**（或 job_id / 通信域唯一标识）生成一块共享区域，例如用 POSIX shm 或 `mmap` 的命名段。
- 布局可设计为（具体字节与对齐可按需细化）：

```
+------------------+------------------------------------------+------------------+
| 版本/序列号      |  Per-rank 统计槽 (rank 0..N-1)           |  参数表          |
| (rank 0 写)      |  (每个 rank 写自己的槽)                  |  (rank 0 写)     |
+------------------+------------------------------------------+------------------+
```

- **Per-rank 槽**：每个 rank 在 SyncStream 时写入自己槽位：例如 `op_key`、`fast_time`、`pcie_time`、`fast_bytes`、`pcie_bytes`、`success` 等（与当前 `ExecStat` + 识别 op 的信息一致）。
- **参数表**：rank 0 在“聚合完本轮统计”后，用当前 controller 更新 param_cache，再把需要暴露给其他 rank 的 param 表（或序列化后的 blob）写入共享区；可带版本号，其他 rank 读时检查版本避免读到半写。

### 3.3 流程（与现有 hook 的衔接）

1. **Collective 入口（如 AllReduce/AllGather）**
   - 每个 rank 照常通过 raw_comm 找到 `CommDomain`。
   - **从共享内存读取当前参数表**（或“当前参数表版本 + 若更新则拷贝到本地 param_cache”），用这份**统一**的 param 做 Lookup / SuggestAlpha / CreatePlan。  
   - 若采用“rank 0 单点写”：只有 rank 0 的进程上才需要真正执行 controller 逻辑；其他 rank 的 `domain->param_cache` 可视为共享区的只读镜像（或直接每次从共享区读）。

2. **Collective 执行**
   - 与现有一致：fast path 用用户 stream，PCIe 用 domain 的 pcie_stream；计时事件挂在各自 stream 上，**不**在 collective 内做 Sync。

3. **SynchronizeStream（aclrtSynchronizeStream / cudaStreamSynchronize）**
   - 每个 rank：取本进程的 pending，sync 本进程的 fast/pcie 计时，得到本 rank 的 `ExecStat`。
   - **每个 rank 只把自己的统计写进共享区中自己的槽位**（不在这里做跨 rank 聚合）。
   - **Rank 0**：在写入自己的槽位之后，需要**知道“所有 rank 都已写完本轮的统计”**，再执行：
     - 从共享区读取所有 rank 的槽位；
     - 聚合成全局 stat（如 `fast_time = max(rank_i.fast_time)`，`pcie_time = max(rank_i.pcie_time)`，bytes 等可按需取一致值或 max）；
     - 调用 `controller->Update(op_key, global_stat, param_cache)`；
     - 将更新后的 param 表（及可选版本号）写回共享区。
   - “所有 rank 都已写完”的实现方式可选：
     - **方式 1**：不在此处做显式 barrier，而是把 rank 0 的“读全槽 + 聚合 + 更新”移到**下一次 collective 的入口**：那时可以认为上一轮 collective 的所有 SyncStream 都已完成，rank 0 再读共享区、聚合、更新、写回，然后所有 rank（含 rank 0）在这次 collective 开始时读最新参数。这样不需要在 SyncStream 里做进程间同步，只依赖“collective 调用顺序”的语义。
     - **方式 2**：在 SyncStream 里用一把共享的“轮次锁”或原子计数器：每个 rank 写完槽后把计数器 +1，rank 0 轮询直到计数等于 world_size 再聚合写回（需要小心 livelock 和超时）。

推荐先用**方式 1**（rank 0 在**下一次 collective 入口**做聚合与写回），逻辑简单，且与现有“每个 rank 在 collective 入口读 param”自然结合。

### 3.4 Rank 0 的识别

- 在 `CommInit`（ncclCommInitRank / HcclCommInitRank）时，每个进程已知自己的 `rank` 和 `nranks`，可把 `rank == 0` 存到 domain 或 DomainManager 中，用于：
  - 仅 rank 0 在“下一次 collective 入口”执行：读共享区所有槽 → 聚合 → Update → 写回参数表。
  - 仅 rank 0 写共享区中的“参数表”和“版本号”；其他 rank 只写自己的统计槽、只读参数表。

### 3.5 共享内存的创建与命名

- **创建时机**：可在 `RegisterRawComm` / `GetOrCreateDomainByKey` 时，根据 `CommDomainKey` 生成唯一名称（例如 hash(key) + 前缀），然后创建或挂接该共享段。
- **命名**：例如 POSIX `shm_open("/ampccl_<hash>", O_CREAT|O_RDWR, 0666)` + `ftruncate` + `mmap`；或使用 `boost::interprocess` / 其他跨进程共享库。需要保证同一 job 内各 rank 使用**同一名称**（例如通过环境变量传入 job_id 或 key 的字符串形式）。
- **生命周期**：与通信域一致；最后一个使用该 key 的 rank 销毁时可 unlink（需注意多进程并发 unlink 的语义）。

### 3.6 与现有代码的衔接点

- **DomainManager / CommDomain**：  
  - 每个 rank 仍保留“本进程的”CommDomain 和 domain 指针（用于 fast/pcie 执行、stream、计时器等）；  
  - 但 **param_cache 的来源**改为：优先从共享内存读取；若使用“rank 0 在下一 collective 入口更新”，则 rank 0 在 collective 入口先读全槽、聚合、Update、写回，再让所有 rank（含 rank 0）从共享区加载到本次 collective 使用的 param。
- **stream_sync.cc（OnStreamSynchronized）**：  
  - 仍按 stream 取 pending、sync 计时、得到本 rank 的 ExecStat；  
  - **不再**在本进程调用 `domain->controller->Update(...)`；  
  - 改为：把本 rank 的 stat 写入共享区中**本 rank 的槽位**；  
  - Rank 0 的“聚合 + Update + 写回”按上面约定移到**下一次 collective 入口**（或在 SyncStream 里用方式 2，任选其一）。
- **VirtualCollective（Collective 入口）**：  
  - 在 Lookup / SuggestAlpha 之前，先从共享内存**读取**当前参数表到本地（或到 domain 的 param_cache 镜像）；  
  - 若当前进程是 rank 0，且共享区中存在“上一轮”各 rank 的统计，则先做：聚合 → controller->Update → 写回参数表；然后再用最新参数做本次 Lookup / SuggestAlpha / CreatePlan。

这样，**测量的是整体集合通信时间（max over ranks）**，**8 个 rank 看到的参数表是同一份**，且**只有 rank 0 修改**（通过共享内存写回），从而满足你的需求。若你确认采用方案 A，下一步可以在代码里加上“共享内存段创建 / 读写 / rank 0 聚合”的接口与占位实现，再逐步替换现有 param_cache 的读写与 Update 调用点。

## 4. 方案 B 简述（用现有通信做归约，无共享内存）

- 在 SyncStream 之后（或在下一次 collective 的入口），用**同一 communicator** 做两次小 buffer 的 AllReduce（例如 max(fast_time)、max(pcie_time)），使所有 rank 得到相同的全局 stat。
- 若 controller 的 Update 与 Suggest 是**确定性**的，则所有 rank 用同一 global stat 各自执行一次 `controller->Update(...)`，会得到相同的新 alpha/param；这样每个 rank 的 param_cache 可保持本地，无需共享内存。
- 代价：每次 SyncStream（或每次 collective）需要多 1–2 次小型 AllReduce，增加延迟与实现复杂度（需避免与用户 collective 死锁、保证 stream/comm 使用顺序）。适合无法使用共享内存的环境。

## 5. 小结

- **问题**：多进程下每 rank 独立计时、独立维护参数，会导致参数不一致且无法用“整体”时间调参。
- **目标**：整体时间（如 max over ranks）+ 所有 rank 共用一份参数表 + 仅 rank 0 修改。
- **推荐**：先用**共享内存 + rank 0 在下一 collective 入口聚合并写回**的方式实现单源真相与单点更新；若后续有“无共享内存”的需求，再考虑方案 B（AllReduce 全局 stat + 确定性 Update）作为补充。

如果你愿意按方案 A 实现，我可以再根据你当前的 `ParamCache`、`ExecStat`、`OpKey` 和 collective 入口/stream_sync 的调用顺序，写一版具体的共享内存布局和 C++ 接口草稿（例如 `ShmParamStore` 的读写 API 和与 DomainManager 的衔接点）。
