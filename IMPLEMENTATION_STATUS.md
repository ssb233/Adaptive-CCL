# 实现状态

## 已完成

根据 `design.md` 的设计，已完成以下 C++ 代码架构：

### ✅ 模块结构

1. **common/** - 通用模块
   - `op_key.h`: 操作键定义（用于参数缓存查找）
   - `config.h`: 配置管理（环境变量读取）

2. **telemetry/** - 遥测模块
   - `timer.h`: 计时器（支持 CUDA events 和 CPU timers）
   - `stats.h`: 执行统计结构

3. **cache/** - 参数缓存
   - `param_cache.h`: 学习表实现（per-op, per-size）

4. **backend/** - 后端抽象层
   - `backend_base.h`: 后端基类模板
   - `fast_backend.h/cc`: 快速后端（NCCL/HCCL）接口和占位实现
   - `pcie_backend.h/cc`: PCIe 后端接口和占位实现

5. **controller/** - 自适应控制器
   - `algo_base.h`: 算法基类
   - `algo_tcp.h`: TCP-style AIMD 算法
   - `algo_dcqcn.h`: DCQCN-style 算法
   - `algo_factory.h`: 算法工厂（通过环境变量选择）
   - `controller.h`: 控制器主类

6. **core/** - 核心模块
   - `domain.h`: 通信域定义
   - `domain_manager.h`: 域管理器
   - `planner.h`: 拆分规划器
   - `virtual_collective.h`: 虚拟集合通信层（核心执行逻辑）

7. **hook/** - LD_PRELOAD 钩子
   - `nccl_hook.cc`: NCCL API 拦截
   - `hccl_hook.cc`: HCCL API 拦截

8. **构建系统**
   - `CMakeLists.txt`: CMake 构建配置

## 关键特性

### 环境变量配置

算法选择通过 `AMPCCL_ALGO` 环境变量控制：
- `tcp`: TCP-style AIMD 算法（默认）
- `dcqcn`: DCQCN-style 算法
- `static`: 静态固定比例算法

其他配置项：
- `AMPCCL_MIN_CHUNK_SIZE`: 最小分块大小
- `AMPCCL_MIN_MSG_SIZE`: 启用 PCIe 的最小消息大小
- `AMPCCL_ENABLE_PCIE`: 启用/禁用 PCIe
- `AMPCCL_DEBUG`: 调试模式

### PCIe 后端集成

PCIe 集合通信 API 将通过 `#include` 方式直接调用，格式与 NCCL/HCCL 类似。在 `pcie_backend.cc` 中预留了集成位置。

## 待完善

1. **Backend 实际实现**：
   - `fast_backend.cc`: 需要调用真实的 NCCL/HCCL API
   - `pcie_backend.cc`: 需要 include PCIe CCL 头文件并调用 API

2. **Domain 信息提取**：
   - 从 NCCL/HCCL communicator 提取 topology 信息
   - 实现 topology_hash 计算

3. **完整的集合操作**：
   - ReduceScatter 完整实现
   - Broadcast 完整实现
   - 其他集合操作

4. **错误处理增强**：
   - 更完善的错误码转换
   - 失败回退机制

5. **测试和示例**：
   - 单元测试
   - 集成测试
   - 使用示例

## 文件统计

- 头文件 (.h): 15 个
- 源文件 (.cc): 4 个
- 构建文件: 1 个 (CMakeLists.txt)
- 文档: 3 个 (design.md, ARCHITECTURE.md, IMPLEMENTATION_STATUS.md)

总计: 23 个文件
