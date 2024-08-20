#include <iostream>
#include <cuda_runtime.h>

__global__ void sgemm(int M, int N, int K, float *A, float *B, float *C) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < M && n < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[m * K + i] * B[i * N + n];
        }
        C[m * N + n] = sum;
    }
}

int main() {
    int M = 1024; // Example value
    int N = 1024; // Example value
    int K = 1024; // Example value

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    // Initialize host matrices
    // (Omitted for brevity; you'd typically fill these matrices with data here)

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy host matrices to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch kernel
    sgemm<<<grid, block>>>(M, N, K, d_A, d_B, d_C);

    // Record the stop event
    cudaEventRecord(stop);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "Time taken by kernel: " << elapsedTime << " milliseconds" << std::endl;

    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute FLOPS
    float flops = 2.0f * M * N * K;
    float timeInSeconds = elapsedTime / 1000.0f; // Convert milliseconds to seconds
    float gflops = (flops / 1e9f) / timeInSeconds; // FLOPS to GFLOPS

    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
