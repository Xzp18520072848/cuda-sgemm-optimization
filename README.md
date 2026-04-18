# CUDA SGEMM Optimization

本项目基于 CUDA 实现并优化了单精度通用矩阵乘法 (SGEMM)，在 NVIDIA RTX 5070 上获得了 **2.6 倍** 的加速。

## 核心技术
- **共享内存分块**：将全局内存数据缓存至共享内存，降低访存延迟。
- **参数自动调优**：探索并确定了最优的 `TILE_SIZE` 和 `THREAD_DIM` 配置。

## 性能对比 (M=N=K=1024)
| 版本 | 耗时 | 加速比 |
| :--- | :--- | :--- |
| 朴素版 | ~1.42 ms | 1.0x |
| 优化版 | ~0.54 ms | **2.6x** |

## 文件说明
- `sgemm_shared.cu`：包含朴素版和共享内存优化版的实现源码。
- `add_kernel.cu`：自定义 CUDA 向量加法算子示例。

## 运行环境
- Ubuntu 22.04 (WSL2)
- CUDA 13.0
- NVIDIA RTX 5070 Laptop GPU
