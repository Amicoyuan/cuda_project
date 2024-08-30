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
#define M_TILE 128
#define K_TILE 32
#define N_TILE 256
#define THREAD_PER_BLOCK 256

// 向量化访存，预取 4 * 4B ,  8 * 2B, 即 8个half
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

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

__global__ void sgemm_warp_tensor_op(const int M, const int N, const int K,
    void * __restrict__ d_A, void * __restrict__ d_B, void * __restrict__ d_C
    ) {
    

    half * a = reinterpret_cast<half *>((void *)d_A);
    half * b = reinterpret_cast<half *>((void *)d_B);
    half * c = reinterpret_cast<half *>((void *)d_C);

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;

    // 右移 5位 相当于 tid / 32
    int wid = tid >> 5;  // warp_tid  当前有8个warp

    // 每个 warp 算的 C_tile 为 [M_64, N_64]

    const int APAD = 8;  // pad 避免 bank conflict
    const int BPAD = 8;
    

    // NV 共有 32个 bank , 一个 bank 4B。 总共 128 B
    // float4 是 4 * 4 = 16B，
    // 但是 FLOAT4 是8个half,  
    // 这里就需要 pad 4 * 4B = 16B, 刚好对应者 2 * 8个half = 16B 
    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BK][BN + BPAD];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_c[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }
    /*
    tid = 0-3   load_a_smem_m 0
    tid = 4-7   load_a_smem_m 2
    tid = 124-127 load load_a_smem_m 62
    tid = 252-255 load load_a_smem_m 126
    */
    int load_a_smem_m = (tid >> 2) << 1;    // 除 4 * 2
    /*
    // 每个线程对应拿8个half
    tid     load_a_smem_k
    0       0
    1       8
    2      16
    3      24
    */
    int load_a_smem_k = (tid &  3) << 3;    // 与操作 再 * 8 = tid 的最后两位乘以 8

    /*
    tid     load_b_smem_k
    0       0
    1       0
    ...
    31      0
    32      4
    ...
    63      4
    64      8
    ...
    95      8
    96      12
    ...
    127     12
    128     16
    ...
    159     16
    160     20
    ...
    191     20
    192     24
    ...
    223     24
    224     28
    ...
    255     28
    */
    int load_b_smem_k = (tid >> 5) << 2; // 除以32 * 4
    /*
    tid     load_b_smem_n
    0       0
    1       8
    2      16
    3      24
    ...
    27     216
    28     224
    29     232
    30     240
    31     248
    32      0
    33     16
    ...
    61     248
    62      0
    63     248
    64      0
    ...
    255   248
    */
    int load_b_smem_n = (tid & 31) << 3; // 这个表达式相当于将 tid 的最低5位乘以 8

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid &  1;
    int comp_c_frag_n = wid >> 1;

    for (int bk = 0; bk < K / BK; bk++) {
        // 每 4 个线程拿 A [M_2, K_32]
        FLOAT4(s_a[load_a_smem_m    ][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr        ]);
        FLOAT4(s_a[load_a_smem_m + 1][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr +     K]);

        // 每 32 个线程  拿 B [ K_4, N_256 ]
        FLOAT4(s_b[load_b_smem_k    ][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr        ]);
        FLOAT4(s_b[load_b_smem_k + 1][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr +     N]);
        FLOAT4(s_b[load_b_smem_k + 2][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr + 2 * N]);
        FLOAT4(s_b[load_b_smem_k + 3][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr + 3 * N]);
        


        // 偏移指正 下轮 K迭代
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        __syncthreads();
        

        // warp tensor op 读 smem
        wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64     ][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64     ][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0][0], &s_b[ 0][comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[ 0][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[ 0][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[ 0][comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);



        // A行B列  增加数据复用
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        __syncthreads();
    }


    // 写回C
    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
        }
    }
}


int main() {
    int M = 4096; // Example value
    int N = 8192; // Example value
    int K = 8192; // Example value

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