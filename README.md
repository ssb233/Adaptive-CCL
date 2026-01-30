# AMP-CCL 代码架构说明

本文档说明根据 `design.md` 实现的 C++ 代码架构。

## 目录结构

```
libampccl/
├── hook/                    # LD_PRELOAD 钩子层
│   ├── nccl_hook.cc        # NCCL API 拦截
│   └── hccl_hook.cc        # HCCL API 拦截
│
├── core/                   # 核心模块
│   ├── virtual_collective.h    # 虚拟集合通信层（核心执行逻辑）
│   ├── domain.h                 # 通信域定义
│   ├── domain_manager.h         # 域管理器（communicator → domain 映射）
│   └── planner.h                # 拆分规划器（split & drop-back 逻辑）
│
├── controller/             # 自适应控制器
│   ├── controller.h            # 控制器主类
│   ├── algo_base.h             # 算法基类
│   ├── algo_factory.h          # 算法工厂（通过环境变量选择算法）
│   ├── algo_tcp.h              # TCP-style AIMD 算法
│   └── algo_dcqcn.h            # DCQCN-style 算法
│
├── backend/                # 后端抽象层
│   ├── backend_base.h          # 后端基类模板
│   ├── fast_backend.h          # 快速后端（NCCL/HCCL）
│   ├── fast_backend.cc         # 快速后端实现
│   ├── pcie_backend.h          # PCIe 后端
│   └── pcie_backend.cc         # PCIe 后端实现
│
├── cache/                  # 参数缓存
│   └── param_cache.h           # 学习表（per-op, per-size）
│
├── telemetry/              # 遥测模块
│   ├── timer.h                 # 计时器（CUDA events / CPU timers）
│   └── stats.h                 # 执行统计
│
└── common/                 # 通用模块
    ├── op_key.h                # 操作键（用于缓存查找）
    └── config.h                # 配置（环境变量读取）
```

## 核心设计要点

### 1. 环境变量配置

算法选择通过环境变量 `AMPCCL_ALGO` 控制：

```bash
export AMPCCL_ALGO=tcp      # 使用 TCP-style AIMD 算法（默认）
export AMPCCL_ALGO=dcqcn    # 使用 DCQCN-style 算法
export AMPCCL_ALGO=static   # 使用静态固定比例算法
```

其他环境变量：
- `AMPCCL_MIN_CHUNK_SIZE`: 最小分块大小（默认 4096 字节）
- `AMPCCL_MIN_MSG_SIZE`: 启用 PCIe 的最小消息大小（默认 8192 字节）
- `AMPCCL_ENABLE_PCIE`: 启用/禁用 PCIe 后端（默认 1）
- `AMPCCL_LOG_LEVEL`: 日志级别 `0`/`off`、`1`/`error`、`2`/`warn`、`3`/`info`、`4`/`debug`；测试融合 NCCL/HCCL 时建议设为 `info` 或 `3`

### 2. PCIe 后端集成

PCIe 集合通信 API 将通过 `#include` 方式直接调用，API 格式与 NCCL/HCCL 类似：

```cpp
// 在 pcie_backend.cc 中：
// #include "pcieccl.h"  // 或供应商提供的头文件
// pciecclResult_t pciecclAllReduce(...)
```

### 3. 执行流程

1. **Hook 层**：拦截 NCCL/HCCL API 调用
2. **Domain 查找**：根据 communicator 获取或创建 CommDomain
3. **Virtual Collective**：
   - 构建 OpKey
   - 从 ParamCache 查找参数
   - Controller 建议 alpha（拆分比例）
   - Planner 创建拆分计划
   - 并行启动 Fast 和 PCIe 后端
   - 测量执行时间
   - Controller 更新算法状态
   - 更新 ParamCache

### 4. 模板化后端

使用模板特化实现零开销抽象：

```cpp
template<typename Backend>
class BackendBase { ... };

// 特化
template<> class BackendBase<FastBackend> { ... };
template<> class BackendBase<PCIeBackend> { ... };
```

### 5. 算法工厂模式

通过 `AlgoFactory::Create()` 根据环境变量创建算法实例：

```cpp
auto algo = AlgoFactory::Create(domain);  // 读取 AMPCCL_ALGO
```

### 6. 日志

通过**全局日志级别**控制输出（环境变量 `AMPCCL_LOG_LEVEL` 或代码内 `ampccl::SetLogLevel()`），级别：`OFF`/`ERROR`/`WARN`/`INFO`/`DEBUG`。日志输出到 stderr，包括：
- 两个 CCL 任务执行**前**：op、bytes、alpha、use_pcie、fast_bytes、pcie_bytes；
- 两个 CCL 任务执行**后**：fast_time、pcie_time、划分参数等；
- 从 rawComm **创建新 Comm** 或 **查到已有 Comm** 时：world_size、topology_hash。

详见 [BUILD.md](BUILD.md)。

## 构建

见 **[BUILD.md](BUILD.md)**。简要：

```bash
./scripts/build.sh
# 产物: build/libampccl.so
```

生成的 .so 可用于 LD_PRELOAD：

```bash
export AMPCCL_LOG_LEVEL=info
export LD_PRELOAD=/path/to/build/libampccl.so
./your_application
```

## 待实现部分

1. **Backend 实现**：
   - `fast_backend.cc`: 实际调用 NCCL/HCCL API
   - `pcie_backend.cc`: 实际调用 PCIe CCL API（待 include 头文件）

2. **Domain 信息提取**：
   - 从 NCCL/HCCL communicator 提取 topology 信息
   - 计算 topology_hash

3. **完整的集合操作**：
   - ReduceScatter
   - Broadcast
   - 其他操作

4. **错误处理**：
   - 更完善的错误码转换
   - 失败回退机制

## 扩展性

- **新算法**：继承 `AdaptiveAlgo` 并在 `AlgoFactory` 中注册
- **新后端**：创建新的 Backend 特化
- **新集合操作**：在 `VirtualCollective` 中添加新方法
