# CUDA SGEMM Optimization

本项目基于 CUDA 实现并优化了单精度通用矩阵乘法 (SGEMM)，通过共享内存分块 (Tiling) 和参数调优，在 RTX 5070 上相比朴素版本获得了 **2.6倍** 的加速。

## 核心优化技术
- **共享内存分块**：将全局内存数据缓存至共享内存，大幅降低访存延迟。
- **Bank Conflict 消除**：通过引入 Padding，减少共享内存的存储体冲突。
- **参数自动调优**：探索并确定了最优的 `TILE_SIZE` 和 `THREAD_DIM` 配置。
- **Nsight Compute 性能分析**：使用 NVIDIA Nsight Compute 精准定位性能瓶颈并验证优化效果。

## 性能对比 (M=N=K=1024)
| 版本 | 耗时 (ms) | 加速比 |
|---|---|---|
| 朴素版 | ~1.42 | 1.0x |
| 优化版 (v2) | ~0.54 | **2.6x** |

## 文件说明
- `sgemm_shared.cu`: 包含朴素版和优化版 (TILE_SIZE=64) 的实现源码。
- `sgemm_report.ncu-rep`: Nsight Compute 生成的完整性能分析报告。
