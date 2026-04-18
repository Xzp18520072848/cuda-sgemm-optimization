#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 64          // 共享内存分块大小
#define THREAD_DIM 16         // 线程块维度
#define ELEM_PER_THREAD 4     // 每个线程在每个维度上处理的元素数

// 朴素版矩阵乘法（用于对比）
__global__ void sgemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 工业标准共享内存版：TILE_SIZE=64，线程块 16x16，单线程处理 4x4 输出
__global__ void sgemm_shared_v2(const float* A, const float* B, float* C, int M, int N, int K) {
    // 共享内存：存放 A 和 B 的子块（加 Padding 避免 bank conflict）
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // 该线程负责的 4x4 输出区域的起始行和列
    int row_start = by * TILE_SIZE + ty * ELEM_PER_THREAD;
    int col_start = bx * TILE_SIZE + tx * ELEM_PER_THREAD;

    // 每个线程维护 4x4 个累加器（寄存器数组）
    float sum[ELEM_PER_THREAD][ELEM_PER_THREAD] = {{0.0f}};

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // ----- 加载 A 子块（使用 float4 向量化）-----
        for (int i = 0; i < ELEM_PER_THREAD; i++) {
            int a_row = row_start + i;
            int a_col_start = tile * TILE_SIZE + tx * ELEM_PER_THREAD;

            // 指向共享内存目标位置（每个线程负责的 4 列）
            float* As_target = &As[ty * ELEM_PER_THREAD + i][tx * ELEM_PER_THREAD];

            if (a_row < M && a_col_start + 3 < K) {
    const float4* A_ptr = reinterpret_cast<const float4*>(A + a_row * K + a_col_start);
    float4 val = *A_ptr;
    As_target[0] = val.x;
    As_target[1] = val.y;
    As_target[2] = val.z;
    As_target[3] = val.w;
}
        }

        // ----- 加载 B 子块（保持原样，已经是合并访问）-----
        for (int i = 0; i < ELEM_PER_THREAD; i++) {
            for (int j = 0; j < ELEM_PER_THREAD; j++) {
                int b_row = tile * TILE_SIZE + ty * ELEM_PER_THREAD + i;
                int b_col = col_start + j;
                int bs_row = ty * ELEM_PER_THREAD + i;
                int bs_col = tx * ELEM_PER_THREAD + j;
                Bs[bs_row][bs_col] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
            }
        }
        __syncthreads();

        // 在共享内存上计算部分点积
        for (int k = 0; k < TILE_SIZE; k++) {
            for (int i = 0; i < ELEM_PER_THREAD; i++) {
                for (int j = 0; j < ELEM_PER_THREAD; j++) {
                    sum[i][j] += As[ty * ELEM_PER_THREAD + i][k] * Bs[k][tx * ELEM_PER_THREAD + j];
                }
            }
        }

        __syncthreads();
    }

    // 将结果写回全局内存
    for (int i = 0; i < ELEM_PER_THREAD; i++) {
        for (int j = 0; j < ELEM_PER_THREAD; j++) {
            int row = row_start + i;
            int col = col_start + j;
            if (row < M && col < N) {
                C[row * N + col] = sum[i][j];
            }
        }
    }
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    const size_t bytes_A = M * K * sizeof(float);
    const size_t bytes_B = K * N * sizeof(float);
    const size_t bytes_C = M * N * sizeof(float);

    float *h_A, *h_B, *h_C_naive, *h_C_shared;
    float *d_A, *d_B, *d_C_naive, *d_C_shared;

    h_A = (float*)malloc(bytes_A);
    h_B = (float*)malloc(bytes_B);
    h_C_naive = (float*)malloc(bytes_C);
    h_C_shared = (float*)malloc(bytes_C);

    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C_naive, bytes_C);
    cudaMalloc(&d_C_shared, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    // 配置线程块和网格
    dim3 block(THREAD_DIM, THREAD_DIM);  // 16x16 = 256 线程（合法！）
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    std::cout << "TILE_SIZE = " << TILE_SIZE << std::endl;
    std::cout << "Thread block: (" << block.x << ", " << block.y << ") = "
              << block.x * block.y << " threads" << std::endl;
    std::cout << "Each thread computes " << ELEM_PER_THREAD << "x" << ELEM_PER_THREAD
              << " = " << ELEM_PER_THREAD * ELEM_PER_THREAD << " outputs" << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // 测速：朴素版
    cudaEventRecord(start);
    sgemm_naive<<<grid, block>>>(d_A, d_B, d_C_naive, M, N, K);
    cudaError_t err_naive = cudaGetLastError();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_naive;
    cudaEventElapsedTime(&ms_naive, start, stop);
    if (err_naive != cudaSuccess) {
        std::cerr << "[朴素版] Kernel launch error: " << cudaGetErrorString(err_naive) << std::endl;
    }

    // 测速：共享内存优化版 v2
    cudaEventRecord(start);
    sgemm_shared_v2<<<grid, block>>>(d_A, d_B, d_C_shared, M, N, K);
    cudaError_t err_shared = cudaGetLastError();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_shared;
    cudaEventElapsedTime(&ms_shared, start, stop);
    if (err_shared != cudaSuccess) {
        std::cerr << "[共享内存版 v2] Kernel launch error: " << cudaGetErrorString(err_shared) << std::endl;
    }

    std::cout << "朴素版耗时: " << ms_naive << " ms\n";
    std::cout << "共享内存版 v2 耗时: " << ms_shared << " ms\n";
    if (err_naive == cudaSuccess && err_shared == cudaSuccess) {
        std::cout << "加速比: " << ms_naive / ms_shared << "x\n";
    } else {
        std::cout << "由于 kernel 启动失败，加速比无意义。\n";
    }

    // 清理
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_naive); cudaFree(d_C_shared);
    free(h_A); free(h_B); free(h_C_naive); free(h_C_shared);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}