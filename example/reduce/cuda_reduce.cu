#include <cuda_runtime.h>
#include <iostream>



// kernel -> blockreduce -> warp_reduce
__device__ float warp_reduce_sum(float value) {
    const unsigned int warpSize = 32;
    unsigned int laneId = threadIdx.x % warpSize;

    // 使用 warp 内部的 shuffle 操作进行归约
    for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
        float neighbor = __shfl_xor_sync(0xFFFFFFFF, value, offset);
        value += neighbor;
    }
    return value;
}

__device__ void block_reduce_sum(float* sharedMem, float value) {
    const unsigned int warpSize = 32;
    unsigned int warpId = threadIdx.x / warpSize;
    unsigned int laneId = threadIdx.x % warpSize;

    // 每个 warp 的归约结果存储到共享内存中
    float warpSum = warp_reduce_sum(value);
    if (laneId == 0) {
        sharedMem[warpId] = warpSum;
    }
    __syncthreads();

    // 如果是块内第一个 warp，计算块内总和
    if (warpId == 0) {
        if (threadIdx.x < warpSize) {
            float blockSum = 0.0f;
            for (unsigned int i = 0; i < (blockDim.x / warpSize); ++i) {
                blockSum += sharedMem[i];
            }
            if (threadIdx.x == 0) {
                sharedMem[0] = blockSum; // 存储块的总和
            }
        }
    }
}

__global__ void kernel(float *data, float *result, int size) {
    extern __shared__ float sharedMem[];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float value = data[idx];
        block_reduce_sum(sharedMem, value);
    }

    // 将每个块的结果写入全局内存
    if (threadIdx.x == 0) {
        result[blockIdx.x] = sharedMem[0]; // 块的总和
    }
}

int main() {
    const int numElements = 1024;
    float *h_data = new float[numElements];
    float *h_result = new float[numElements / 32];
    float *d_data, *d_result;

    // 初始化主机数据
    for (int i = 0; i < numElements; ++i) {
        h_data[i] = 1.0f; // 或者其他你想要的值
    }

    cudaMalloc(&d_data, numElements * sizeof(float));
    cudaMalloc(&d_result, (numElements / 32) * sizeof(float));

    cudaMemcpy(d_data, h_data, numElements * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernel，假设每个块有 32 个线程
    kernel<<<numElements / 32, 32, 32 * sizeof(float)>>>(d_data, d_result, numElements);

    cudaMemcpy(h_result, d_result, (numElements / 32) * sizeof(float), cudaMemcpyDeviceToHost);

    // 输出结果
    float finalSum = 0.0f;
    for (int i = 0; i < numElements / 32; ++i) {
        finalSum += h_result[i];
    }
    std::cout << "Sum: " << finalSum << std::endl;

    cudaFree(d_data);
    cudaFree(d_result);
    delete[] h_data;
    delete[] h_result;

    return 0;
}
