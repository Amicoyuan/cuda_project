#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include "conv2d.h"


__global__ void implgemm(param_t param)
{   


    // 这里
    // output tensor: N * oh * ow * k
    // block实际上也是切了 K 与   N * on * ow
    // 而block内的线程块才是处理 矩阵计算中的K维度，即 C * S * R
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if(x >=param.Oh * param.Ow || y>= param.k || z>= param.n)
    {
        return;
    }


    // 当前线程处理的数据点在oh,ow上的坐标
    int posOh = x /param.Ow;
    int posOw = x %param.Ow;

    int posh_ori = posOh * param.u - param.p;
    int posw_ori = posOw * param.v - param.q;

    float sum = 0.0f;

    int inOffset = z * param.c * param.h * param.w + posh_ori * param.w + posw_ori;
    int weiOffset = y * param.c * param.r * param.s;
    int inChannelOffset = param.h * param.w;
    int weightChannelOffset = param.r * param.s;

    for(int i =0;i<param.r*param.s*param.c;i++) {

        int weiOffsetTmp = i;
        int curC = i / (param.r  * param.s);
        int curR = (i % (param.r * param.s)) / param.s;
        int curS = (i % (param.r * param.s)) % param.s;
        int curH = posh_ori + curR;
        int curW = posw_ori + curS;
        int inOffsetTmp = curC * inChannelOffset + curR * param.w + curS;
        if(curH >=0 && curW >=0 && curW <param.w && curH < param.h)
        {
            sum+= param.weight[weiOffset + weiOffsetTmp] * param.input[inOffset + inOffsetTmp];
        }

    }
    int outOffset = z * param.k * param.Oh * param.Ow + y * param.Oh * param.Ow + x;
    param.output[outOffset] = sum;
}




void launch_param(param_t param)
{   
    unsigned int n = param.n;
    unsigned int c = param.c;
    unsigned int h = param.h;
    unsigned int w = param.w;
    unsigned int k = param.k;
    unsigned int r = param.r;
    unsigned int s = param.s;
    unsigned int u = param.u;
    unsigned int v = param.v;
    unsigned int p = param.p;
    unsigned int q = param.q;


    // 坐标映射
    int outh = (h-r + 2 * p) / u +1;
    int outw = (w-s + 2 * q) / v+ 1;

    param.Oh = outh;
    param.Ow = outw;


    int blockx = ((outh * outw +16 -1) /16);   // oh * ow按x分块
    int blocky = (k+ 16-1)/16;  // block 按y分块
    int blockz = n;  // n方向切到z轴去
    int threadx = 16;
    int thready = 16;
    int threadz = 1;
    dim3 block(threadx,thready,threadz);
    dim3 grid(blockx,blocky,blockz);

    implgemm<<<grid,block>>>(param);
}
int main(int argc, char **argv)
{
    // unsigned int n = atoi(argv[1]);
    // unsigned int c = atoi(argv[2]);
    // unsigned int h = atoi(argv[3]);
    // unsigned int w = atoi(argv[4]);
    // unsigned int k = atoi(argv[5]);
    // unsigned int r = atoi(argv[6]);
    // unsigned int s = atoi(argv[7]);
    // unsigned int u = atoi(argv[8]);
    // unsigned int v = atoi(argv[9]);
    // unsigned int p = atoi(argv[10]);
    // unsigned int q = atoi(argv[11]);

    unsigned int n = 1;
    unsigned int c = 3;
    unsigned int h = 32;
    unsigned int w = 32;
    unsigned int k = 64;
    unsigned int r = 3;
    unsigned int s = 3;
    unsigned int u = 1;
    unsigned int v = 1;
    unsigned int p = 1;
    unsigned int q = 1;

    int outh = (h - r + 2 * p) / u + 1;
    int outw = (w - s + 2 * q) / v + 1;
    double M = k;
    double N = n * outh * outw;
    double K = c * r * s;
    double temp = n * outh * outw * 1e-9f;
    double flopsPerConv = temp * M * K * 2.0;
    float *input = (float *)malloc(n * c * h * w * sizeof(float));
    float *weight = (float *)malloc(k * c * r * s * sizeof(float));
    float *bias = (float *)malloc(k * sizeof(float));
    float *output = (float *)malloc(n * k * outh * outw * sizeof(float));
    float *output_host = (float *)malloc(n * k * outh * outw * sizeof(float));

    float *input_device, *weight_device, *bias_device, *output_device;
    cudaMalloc((void **)&input_device, n * c * h * w * sizeof(float));
    cudaMalloc((void **)&weight_device, k * c * r * s * sizeof(float));
    cudaMalloc((void **)&bias_device, k * sizeof(float));
    cudaMalloc((void **)&output_device, n * k * outh * outw * sizeof(float));

    for (int i = 0; i < n * c * h * w; i++)
    {
        input[i] = (rand() % 255) / 255.0;
    }

    for (int i = 0; i < k * c * r * s; i++)
    {
        weight[i] = (rand() % 255) / 255.0;
    }

    for (int i = 0; i < k; i++)
    {
        bias[i] = (rand() % 255) / 255.0;
    }

    for (int i = 0; i < n * k * outh * outw; i++)
    {
        output[i] = 0.0;
        output_host[i] = 0.0;
    }

    cudaMemcpy(input_device, input, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_device, weight, k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_device, bias, k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_device, output, n * k * outh * outw * sizeof(float), cudaMemcpyHostToDevice);

    //Convolution parameter

    param_t param;

    param.input = input_device;
    param.weight = weight_device;
    param.bias = bias_device;
    param.output = output_device;
    param.n = n;
    param.c = c;
    param.h = h;
    param.w = w;
    param.k = k;
    param.r = r;
    param.s = s;
    param.u = u;
    param.v = v;
    param.p = p;
    param.q = q;
    param.Oh = outh;
    param.Ow = outw;
    /********************************** step 2****************************/


    /*******************************warm up and get result************************************/
    launch_param(param);

    cudaMemcpy(output_host, output_device, n * k * outh * outw * sizeof(float), cudaMemcpyDeviceToHost);

    /*******************************cost time test************************************/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    float time_elapsed = 0.0;

    int iternum = 10;
    for (int i = 0; i < iternum; i++)
    {
        launch_param(param);
    }
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // printf("===================start verfiy===================\n");
    // direct_conv2dcpu(input, weight, bias, output, n, c, h, w, k, r, s, u, v, p, q);

    // int error = 0;
    // for (int i = 0; i < n * k * outh * outw; i++)
    // {
    //     if (abs(output_host[i] - output[i]) > getPrecision(output[i]))
    //     {
    //         printf("error, postion:%d, gpuvalue:%f, cpuvalue:%f\n", i, output_host[i], output[i]);
    //         error++;
    //         break;
    //     }
    // }
    // printf("================finish,error:%d=========================\n", error);

    float timePerConv = time_elapsed / iternum;
    double gflops = flopsPerConv / (timePerConv / 1000.0f);
    printf("%2d %2d %2d %2d %d %d %2d\n", n, h, w, c, r, s, k);
    printf("time: %f ms\n", timePerConv);
    printf("Performance :%f GFlops\n",  gflops);
    
    cudaFree(input_device);
    cudaFree(weight_device);
    cudaFree(output_device);

    free(input);
    free(weight);
    free(output);
    free(output_host);

    return 0;
}