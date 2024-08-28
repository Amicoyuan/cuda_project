#include <iostream>
#include <cuda_runtime.h>
#include "mma.h"
#include "cuda_fp16.h"

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32
#define WARP_X 4
#define WARP_Y 2
#define M_TILE 32
#define K_TILE 32
#define N_TILE 64
#define THREAD_PER_BLOCK 256

// 向量化访存，预取 4 * 4B ,  8 * 2B, 即 8个half
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

using namespace nvcuda;

// 在这里我们一个block 分了32个线程
// 一个warp 刚好调度32个线程
// tensor op 的抽象层级刚刚好是 warp level
// 而 cuda core op的抽象则是 thread level

// tensor core 相关的op都在  nvcuda:wmma  命名空间下
// 详情见：https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-description 

/*
A100 SM80 FP16的最高算力是 312 TFLOPS
一共有108个SM 
根据计算能力为 8.0 的 A100 GPU, 每个 SM 最多可以支持 64 个并发 warps, 而对于计算能力为 8.6 的 GPU，每个 SM 可以支持最多 48 个并发 warps
一个SM有4个Tensor core
*/

__global__ void sgemm_warp_tensor_op(int M, int N, int K, void *__restrict__ d_A, void *__restrict__ d_B, void *__restrict__ d_C) {
    
    
    half * A = reinterpret_cast<half *>((void *)d_A);
    half * B = reinterpret_cast<half *>((void *)d_B);
    half * C = reinterpret_cast<half *>((void *)d_C);
    
    const int K_L1 = (K + K_TILE - 1) / K_TILE;
    const int K_L2 = (K_L1 + WMMA_K - 1) / WMMA_K;


    __shared__ half smem_a[M_TILE][K_TILE];
    __shared__ half smem_b[N_TILE][K_TILE];

    // 而我们现在这个例子,  一个block有 256个线程, 共有 8个 warp
    // 每个block算的C_tile 变为 32 * 64
    // 一个warp负责 16 * 16的 C_tile_frag
    // Warp数量过少的话，会导致Warp Occupancy过低，即调度器内的Eligible Warps数量过少，
    // 可能会严重影响调度器内部的指令发射周期，最终影响Kernel的性能
    const int tid = threadIdx.x;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warp_x_id = warp_id % WARP_X;
    const int warp_y_id = warp_id / WARP_X;
    const int warp_row = blockIdx.y * M_TILE;
    const int warp_col = blockIdx.x * N_TILE;
    const int warp_thread_id = threadIdx.x % WARP_SIZE;

    if (warp_row >= M || warp_col >= N) {
        return;
    }


    // 补充一下背景知识，wmma::fragment中声明的frag都在 register file中
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0f);


        // 16 16 16 tensor op
#pragma unroll
    for (size_t i = 0; i < K_L1; i++) {


        // 在这里就要使用shared memory了
        // smem_a[M_TILE][K_TILE] smem_a[32][32]
        if(warp_x_id < 2){
            FETCH_FLOAT4(smem_a[warp_y_id * WMMA_M + warp_thread_id / 2][warp_x_id * 16 + warp_thread_id % 2 * 8]) = 
            FETCH_FLOAT4(A[warp_row * K + i * K_TILE   + warp_y_id * K * WMMA_M + warp_x_id * 16 +  warp_thread_id / 2 * K + warp_thread_id % 2 * 8] );
        }
        // smen_b[N_TILE][K_TILE]  smen_b[64][32] 
        FETCH_FLOAT4(smem_b[warp_x_id * WMMA_N + warp_thread_id / 2][warp_y_id * 16 + warp_thread_id % 2 * 8]) = 
        FETCH_FLOAT4(B[i * K_TILE + warp_col * K + warp_x_id * K * WMMA_N + warp_y_id * 16 + warp_thread_id / 2 * K + warp_thread_id % 2 * 8]);
        for (size_t i = 0; i < K_L2; i++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag;
            // ldm描述连续行（对于行主要布局）或列（对于列主要布局）之间的元素跨度
            // 在这里即为K
            
            // 这里偏移很好理解了
            wmma::load_matrix_sync(A_frag, &smem_a[warp_y_id * WMMA_M][ i * WMMA_K], K_TILE); 
            wmma::load_matrix_sync(B_frag, &smem_b[warp_x_id * WMMA_N][i * WMMA_K], K_TILE);
    
            wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        }
    }
    wmma::store_matrix_sync(C + warp_y_id * WMMA_M * N + warp_row * N + warp_x_id * WMMA_N + warp_col  , C_frag, N, wmma::mem_row_major);
}

int main() {
    int M = 2048; // Example value
    int N = 2048; // Example value
    int K = 2048; // Example value

    // Allocate host memory
    float *h_A_ft32 = new float[M * K];
    float *h_B_ft32 = new float[K * N];
    float *h_C_ft32 = new float[M * N];

    uint16_t *h_A_ft16 = new uint16_t[M * K];
    uint16_t *h_B_ft16 = new uint16_t[K * N];
    uint16_t *h_C_ft16 = new uint16_t[M * N];

    printf("init data....\n");



    printf("Host compute....\n"); 


    printf("Device compute....\n"); 
    
    // Allocate device memory
    void *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(uint16_t));
    cudaMalloc(&d_B, K * N * sizeof(uint16_t));
    cudaMalloc(&d_C, M * N * sizeof(uint16_t));

    // Copy host matrices to device
    cudaMemcpy(d_A, h_A_ft16, M * K * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_ft16, K * N * sizeof(uint16_t), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block(THREAD_PER_BLOCK);


    std::cout << "M_TILE  N_TILE " << M_TILE <<" "<< N_TILE<<std::endl;
    
    dim3 grid( ((N + N_TILE - 1) / N_TILE), ((M + M_TILE - 1) / M_TILE));

    std::cout << "blockDimx  blockDimy " << ((N + N_TILE - 1) / N_TILE) <<" "<< ((M + M_TILE - 1) / M_TILE)<<std::endl;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch kernel
    sgemm_warp_tensor_op<<<grid, block>>>(M, N, K, d_A, d_B, d_C);

    // Record the stop event
    cudaEventRecord(stop);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "Time taken by kernel: " << elapsedTime << " milliseconds" << std::endl;
    std::cout << "M N K " << M <<" "<< N<<" "<< K<<std::endl;
    // Copy result back to host
    cudaMemcpy(h_C_ft16, d_C, M * N * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    // Compute FLOPS
    float flops = 2.0f * M * N * K;
    float timeInSeconds = elapsedTime / 1000.0f; // Convert milliseconds to seconds
    float gflops = (flops / 1e9f) / timeInSeconds; // FLOPS to GFLOPS

    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    // Cleanup
    delete[] h_A_ft32;
    delete[] h_B_ft32;
    delete[] h_C_ft32;
    delete[] h_A_ft16;
    delete[] h_B_ft16;
    delete[] h_C_ft16;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}