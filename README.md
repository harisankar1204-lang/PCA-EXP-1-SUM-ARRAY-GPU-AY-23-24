# PCA: EXP-1  SUM ARRAY GPU
<h3>HARISH S</h3>
<h3>26018007</h3></h3>
<h3>EX. 1</h3>
<h3>18.08.2026</h3>
<h1> <align=center> SUM ARRAY ON HOST AND DEVICE </h3>
PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## AIM:

To perform vector addition on host and device.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1. Initialize the device and set the device properties.
2. Allocate memory on the host for input and output arrays.
3. Initialize input arrays with random values on the host.
4. Allocate memory on the device for input and output arrays, and copy input data from host to device.
5. Launch a CUDA kernel to perform vector addition on the device.
6. Copy output data from the device to the host and verify the results against the host's sequential vector addition. Free memory on the host and the device.

## PROGRAM:
!nvidia-smi

<img width="897" height="436" alt="Screenshot 2026-08-18 102857" src="https://github.com/user-attachments/assets/b98881c6-6feb-4840-9bd1-404888c5a539" />
!pip install -q nvcc4jupyter
%load_ext nvcc4jupyter

<img width="667" height="139" alt="{FA1C7F0B-C356-435B-BD1D-BBFF6E714595}" src="https://github.com/user-attachments/assets/32bad616-227d-49d5-8ad4-59f499cf232f" />
%%cuda

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return tp.tv_sec + tp.tv_usec * 1e-6;
}

// CPU Vector Addition
void vectorAddCPU(float *A, float *B, float *C, int N)
{
    for(int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
}

// GPU Kernel
__global__ void vectorAddGPU(float *A, float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N)
        C[i] = A[i] + B[i];
}

// Compare Results
void checkResult(float *cpu, float *gpu, int N)
{
    for(int i = 0; i < N; i++)
    {
        if(fabs(cpu[i] - gpu[i]) > 1e-5)
        {
            printf("Arrays do not match!\n");
            return;
        }
    }

    printf("Arrays match.\n");
}

int main()
{
    printf("Starting...\n");

    // Device Information
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,0);

    printf("Using Device 0: %s\n", deviceProp.name);

    int N = 1 << 24;

    printf("Vector size %d\n", N);

    size_t bytes = N * sizeof(float);

    // Host Memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *cpuResult = (float*)malloc(bytes);
    float *gpuResult = (float*)malloc(bytes);

    // Initialize Data
    double start = seconds();

    for(int i=0;i<N;i++)
    {
        h_A[i] = rand()%100;
        h_B[i] = rand()%100;
    }

    printf("initialData Time elapsed %f sec\n", seconds()-start);

    // CPU Execution
    start = seconds();

    vectorAddCPU(h_A,h_B,cpuResult,N);

    printf("sumArraysOnHost Time elapsed %f sec\n", seconds()-start);

    // Device Memory
    float *d_A,*d_B,*d_C;

    cudaMalloc(&d_A,bytes);
    cudaMalloc(&d_B,bytes);
    cudaMalloc(&d_C,bytes);

    cudaMemcpy(d_A,h_A,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,bytes,cudaMemcpyHostToDevice);

    int threads = 512;
    int blocks = (N + threads -1)/threads;

    start = seconds();

    vectorAddGPU<<<blocks,threads>>>(d_A,d_B,d_C,N);

    cudaDeviceSynchronize();

    printf("sumArraysOnGPU <<< %d, %d >>> Time elapsed %f sec\n",
            blocks,threads,seconds()-start);

    cudaMemcpy(gpuResult,d_C,bytes,cudaMemcpyDeviceToHost);

    checkResult(cpuResult,gpuResult,N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(cpuResult);
    free(gpuResult);

    return 0;
}




## OUTPUT:

<img width="622" height="170" alt="{CFB07D8A-05AA-4955-98A6-348B9A2D1DF8}" src="https://github.com/user-attachments/assets/0d35ba5c-7a9c-46ce-959b-e3c750f04dff" />


## RESULT:
Thus, Implementation of sum arrays on host and device is done in nvcc cuda using random number.
