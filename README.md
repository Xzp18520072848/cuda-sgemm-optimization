# CUDA SGEMM Optimization & TensorRT Inference Acceleration

本项目包含两个独立的性能优化实验：
1. **CUDA SGEMM 算子优化**：从朴素实现出发，通过共享内存分块、参数自动调优、单线程多输出等优化手段，最终在 RTX 5070 上获得 **2.6 倍** 加速。
2. **TensorRT 推理加速**：对 ResNet-18 进行 FP16 部署，加速比最高 **46x**（稳定约 16~23x）。

---

## 🧩 Part 1: CUDA SGEMM Optimization

### 优化历程
1. **朴素版 (`sgemm_naive`)**：全局内存直接访问，非合并访存导致 `LG Throttle` 严重，耗时 ~1.42 ms。
2. **共享内存分块 (`sgemm_shared`)**：引入 `__shared__` 缓存子块，消除全局内存散乱访问，耗时降至 ~0.54 ms。
3. **参数自动调优**：
   - 固定 `TILE_SIZE=64`，探索线程块配置：
     - `8×8` 线程块，每线程 8×8 输出 → 加速比 2.24x
     - `16×16` 线程块，每线程 4×4 输出 → **加速比 2.6x**（最优）
     - `32×32` 线程块，每线程 2×2 输出 → 加速比 1.9x
   - 发现 `16×16` 配置在并行度与寄存器压力之间达到最佳平衡。

### 核心技术
- **共享内存分块**：将全局内存数据缓存至共享内存，大幅降低访存延迟。
- **Bank Conflict 消除**：共享内存数组加 Padding（`TILE_SIZE+1`）避免存储体冲突。
- **单线程多输出**：每线程处理 4×4 个输出元素，摊销循环开销，提高指令级并行。
- **参数自动调优**：探索 `TILE_SIZE` 与线程块配置的最优组合。

### 性能对比 (M=N=K=1024)
| 版本 | 配置 | 耗时 | 加速比 |
| :--- | :--- | :--- | :--- |
| 朴素版 | 全局内存直接访问 | ~1.42 ms | 1.0x |
| 优化版 v2 | TILE_SIZE=64, 16×16 线程块, 每线程 4×4 输出 | **~0.54 ms** | **2.6x** |

### 相关文件
- `sgemm_shared.cu`：朴素版与共享内存优化版源码。
- `sgemm_shared_backup`：稳定版可执行文件。
- `add_kernel.cu`：自定义 CUDA 向量加法算子示例。

---

## 🚀 Part 2: TensorRT Inference Acceleration

### 实验目标
使用 TensorRT 对 PyTorch 预训练的 ResNet-18 进行 **FP16 推理部署**，对比原生 PyTorch 推理速度。

### 核心步骤
1. **导出 ONNX**：`torch.onnx.export`
2. **构建 TensorRT 引擎**：启用 FP16 标志，针对 RTX 5070 自动调优
3. **性能对比**：循环 100 次取平均耗时

### 性能对比 (输入尺寸: 1×3×224×224)
| 框架 | 精度 | 耗时 | 加速比 (稳定值) | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| PyTorch | FP32 | ~6.9 ms | 1.0x | 基准线 |
| TensorRT | FP16 | **~0.3 ms** | **~23x** | 当前最优配置 |
| TensorRT | INT8 | ~0.61 ms | ~11x | 受限于 RTX 5070 新架构与 TensorRT 10.x 兼容性 |
> **注**：峰值加速比出现在首次引擎构建后，后续稳定在 16~23x。加速比波动源于 GPU 频率动态调整及引擎构建的随机性。

### 相关文件
- `trt_resnet.py`：完整的 PyTorch → ONNX → TensorRT 部署与测速脚本。
- `resnet18.onnx`：导出的 ONNX 中间表示文件。
- `trt_resnet_int8.py`:int8量化测速脚本。
### ⚠️ 技术踩坑：INT8 量化在 RTX 5070 上的兼容性问题
在尝试将 ResNet-18 量化为 INT8 精度时，TensorRT 10.x 版本对 NVIDIA Blackwell 架构（RTX 5070）的 INT8 支持尚不完善。构建引擎时出现大量 `Dequantize [SCALE] has invalid precision Int8, ignored` 警告，导致模型大部分层回退为 FP16 甚至 FP32，额外引入的量化/反量化操作反而使推理耗时增加（0.61ms vs FP16 的 0.3ms）。

**结论**：当前环境下 FP16 是推理加速的最佳选择。INT8 量化需等待 TensorRT 后续版本更新，或采用更激进的手动量化方案（如 QAT）。本实验验证了工业部署中“版本成熟度评估”的重要性，是 AI Infra 日常工作的真实写照。
---

## 🖥️ 运行环境
- Ubuntu 22.04 (WSL2)
- CUDA 13.0
- NVIDIA RTX 5070 Laptop GPU (8GB)
- PyTorch 2.11.0
- TensorRT 10.16.1.11
