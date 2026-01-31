# Adaptive-CCL 编译说明

## 编译产物

- **默认**：生成 **`libampccl.so`**（同时包含 NCCL 与 HCCL 的 hook，供 LD_PRELOAD 使用）。
- **可选**：仅生成 NCCL 或仅生成 HCCL 的 .so，便于与 NCCL/HCCL 源码一起或单独部署。

编译时 **不链接** libnccl.so / libhccl.so，运行时通过 `dlopen`/`dlsym` 调用原始 API，因此：
- 同一份 `libampccl.so` 可用于只链接 NCCL 或只链接 HCCL 的应用；
- 若需参与 NCCL 或 HCCL 的源码编译，可将本仓库作为子模块，只编译对应 hook 的 .so（见下文“解耦编译”）。

---

## 方式一：使用编译脚本（推荐）

```bash
cd Adaptive-CCL
chmod +x scripts/build.sh
./scripts/build.sh
```

产物在 `build/libampccl.so`。

指定构建类型（Debug/Release）：

```bash
./scripts/build.sh -DCMAKE_BUILD_TYPE=Release
```

---

## 方式二：使用 CMake 直接编译

```bash
cd Adaptive-CCL
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

产物：`build/libampccl.so`（当 `BUILD_SHARED_LIBS=ON` 时，默认即生成 .so）。

---

## 计时器与编译选项

计时器按**编译目标**二选一，不会同时启用：

- **NCCL 构建**（`NCCL_ONLY=ON` 或默认且找到 CUDA）：使用 **CUDA Runtime Event** 计时（`cudaEventRecord` 挂到 `cudaStream_t`），需链接 CUDA。
- **HCCL 构建**（`HCCL_ONLY=ON` 或默认且设置 `ASCEND_HOME`）：使用 **ACL Event** 计时（`aclrtRecordEvent` 挂到 `aclrtStream`，参考 pcieccl 的 test_allgather），需 Ascend 头与库。
- 若两者都不可用：使用 **CPU chrono** 作为回退。

构建时 CMake 会打印使用的计时器（"Timer: CUDA events" / "Timer: ACL events" / "Timer: CPU fallback"）。

---

## 解耦编译：仅 NCCL 或仅 HCCL

若希望生成**只含 NCCL hook** 或**只含 HCCL hook** 的 .so（体积更小或与 NCCL/HCCL 源码一起编译时使用）：

**仅 NCCL：**
```bash
./scripts/build.sh -DNCCL_ONLY=ON
# 产物: build/libampccl_nccl.so
```

**仅 HCCL：**
```bash
./scripts/build.sh -DHCCL_ONLY=ON
# 产物: build/libampccl_hccl.so
```

使用方式：

- NCCL 应用：`LD_PRELOAD=/path/to/libampccl.so ./app` 或 `LD_PRELOAD=/path/to/libampccl_nccl.so ./app`
- HCCL 应用：`LD_PRELOAD=/path/to/libampccl.so ./app` 或 `LD_PRELOAD=/path/to/libampccl_hccl.so ./app`

---

## 环境变量（运行时）

| 变量 | 说明 |
|------|------|
| **`AMPCCL_ENABLE`** | **总开关**：是否使用 Adaptive-CCL。设为 `1`/`on`/`true`/`yes` 时走自适应双路（fast + PCIe）；**未设或为其他值时，所有调用直接转发到原始 NCCL/HCCL**，行为与未 LD_PRELOAD 一致。 |
| `AMPCCL_LOG_LEVEL` | 日志级别：`0`/`off`、`1`/`error`、`2`/`warn`、`3`/`info`、`4`/`debug`。 |
| `AMPCCL_ALGO` | 算法：`tcp`、`dcqcn`、`static`。 |
| `AMPCCL_MIN_MSG_SIZE` | 启用 PCIe 的最小消息大小（字节）。 |
| `AMPCCL_ENABLE_PCIE` | `1`/`0` 是否启用 PCIe 路径（仅在 `AMPCCL_ENABLE=1` 时生效）。 |

示例（启用 Adaptive-CCL 并打开 INFO 日志）：

```bash
export AMPCCL_ENABLE=1
export AMPCCL_LOG_LEVEL=info
LD_PRELOAD=/path/to/build/libampccl.so ./your_nccl_or_hccl_app
```

不启用时（走原始 NCCL/HCCL，默认）：

```bash
# 不设置 AMPCCL_ENABLE，或 export AMPCCL_ENABLE=0
LD_PRELOAD=/path/to/build/libampccl.so ./your_nccl_or_hccl_app
```

日志会打印到 **stderr**，包括：
- 两个 CCL 任务执行**前**：op、bytes、alpha、use_pcie、fast_bytes、pcie_bytes；
- 两个 CCL 任务执行**后**：fast_time、pcie_time、fast_bytes、pcie_bytes、成功与否，以及划分参数；
- 从 rawComm **创建新 Comm** 或 **查到已有 Comm** 时：world_size、topology_hash。

---

## 启用 PCIe 后端（pcieccl / PCCL）

若已克隆 [pcieccl](https://github.com/...) 到与 Adaptive-CCL 同级目录（或任意路径）：

1. **先编译 pcieccl（Ascend/HCCL 环境）**

   ```bash
   cd /path/to/pcieccl
   make lib DEVICE=ascend ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
   ```

   产物：`pcieccl/build/lib/libpccl.so`。

2. **再编译 Adaptive-CCL 并链接 PCCL**

   ```bash
   cd /path/to/Adaptive-CCL
   export PCIECCL_ROOT=/path/to/pcieccl
   export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest  # 与 pcieccl 一致
   ./scripts/build.sh
   ```

   或使用一键脚本（默认认为 pcieccl 在 `../pcieccl`）：

   ```bash
   ./scripts/build_with_pcie.sh /path/to/pcieccl
   ```

   CMake 会加入 `-I${PCIECCL_ROOT}/include`、链接 `libpccl.so` 与 Ascend 库，并在 `comm_init.cc` / `pcie_backend.cc` 中启用 PCCL 的 `pcclInit`、`pcclSubmit` 等调用。

3. **不设置 PCIECCL_ROOT 时**  
   PCIe 后端仍参与编译，但不链接 pcieccl，集体通信走 fast 路径，PCIe 调用为桩实现。

---

## 与 NCCL/HCCL 源码一起编译

本仓库设计为**独立编译成 .so 再通过 LD_PRELOAD 注入**，不要求与 NCCL/HCCL 同树编译。

若你希望把 Adaptive-CCL 编进 NCCL 或 HCCL 的构建系统：
1. 将 `libampccl` 头文件与源加入该工程；
2. 仅编译并链接 NCCL 或 HCCL 的 hook 源（`nccl_hook.cc` 或 `hccl_hook.cc`），并链接其余 ampccl 核心实现；
3. 输出仍为 .so，由该工程的 Makefile/CMake 生成即可。

当前 CMake 的 `NCCL_ONLY` / `HCCL_ONLY` 即用于生成只含对应 hook 的 `libampccl_nccl.so` / `libampccl_hccl.so`，便于集成到其它构建中。
