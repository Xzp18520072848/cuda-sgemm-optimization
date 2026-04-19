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
| 框架 | 精度 | 耗时 | 加速比 (稳定值) |
| :--- | :--- | :--- | :--- |
| PyTorch | FP32 | ~6.9 ms | 1.0x |
| TensorRT | FP16 | ~0.3 ms | **~23x** |
| TensorRT (峰值) | FP16 | ~0.27 ms | **~46x** |

> **注**：峰值加速比出现在首次引擎构建后，后续稳定在 16~23x。加速比波动源于 GPU 频率动态调整及引擎构建的随机性。

### 相关文件
- `trt_resnet.py`：完整的 PyTorch → ONNX → TensorRT 部署与测速脚本。
- `resnet18.onnx`：导出的 ONNX 中间表示文件。

---

## 🖥️ 运行环境
- Ubuntu 22.04 (WSL2)
- CUDA 13.0
- NVIDIA RTX 5070 Laptop GPU (8GB)
- PyTorch 2.11.0
- TensorRT 10.16.1.11
