# Adaptive-CCL

一个用于异构链路（高速原生链路 + PCIe）并行的自适应集合通信原型框架，实现了统一后端基类、调度器、简单控制器和示例程序。

## 目录结构
- `include/adaptive_ccl/api`：对外 API 封装（AdaptiveAllReduce / AdaptiveAllGather / AdaptiveInit）
- `include/adaptive_ccl/core`：Dispatcher、BufferManager、ExecStats、基础类型
- `include/adaptive_ccl/controller`：控制器抽象与默认简单比例控制器
- `include/adaptive_ccl/backend`：统一后端基类与 HCCL / NCCL（占位）/ PCIe 后端
- `include/adaptive_ccl/util`：计时、日志、错误工具
- `src`：对应实现
- `examples`：`demo_allreduce.cpp` 示例
- `tests`：简单单元测试骨架

## 构建
```bash
cd Adaptive-CCL
cmake -B build
cmake --build build
ctest --test-dir build   # 运行测试
```

## 快速使用
```cpp
using namespace adaptive_ccl;
auto* ctx = AdaptiveInit();
AdaptiveAllReduce(send, recv, count, DataType::FLOAT32, ReduceOp::SUM, ctx);
AdaptiveFinalize(ctx);
```

## 设计要点
- Dispatcher 查询 Controller 获取拆分比例 α，切分 buffer 并并行下发两个后端（示例实现为串行模拟）。
- Controller 只看执行统计，不依赖具体后端细节，便于更换算法。
- Backend 只做接口翻译，当前用 memcpy/延迟模拟实际通信。

## 后续可扩展
- 替换真实 HCCL/NCCL/PCIe 调用
- 增强控制算法（AIMD、DCQCN 风格、ML）
- 引入 GPU event 计时、拓扑感知、pipeline 拆分等
