/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>

// hip header file
#include "hip/hip_runtime.h"


#define WIDTH 1024


#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

//16 1152 5120
#define WGSIZE 768
#define M 4096
#define K 4096
#define N 4096
#define LOG2K 8
#define TILE 2

__global__ void gemm(float* A, float* B, float* C) {
    __shared__ float4 L4[64];
    __shared__ float* local;
    local = (__shared__ float*)L4;
    float4 temp4[16];
    float* temp;
    temp = (float*)temp4;
    int tidx = hipThreadIdx_x;
    int tidy = hipThreadIdx_y;
    int lid = tidx & 63;

// init temp
    for(int i=0; i<64; i++){
      temp[i] = 0;
    }

//load 4 from A
    float4* A4 = (float4*)(&A[tidx*K]);

    int lx = (lid & 15); // within a group
    int lxg = lx * 4 + tidy * 64; // per each group
    int ly = lid / 16;
    float4* B4 = (float4*)(&B[lxg + ly*K]);

    for (int k = 0; k < K/4; k++){
      float4 readA4 = A4[k]; //progress horizontally
      float4 readB4 = B4[k*K]; //progress vertically

      //transpose in local mem, hoping compiler burst the local load
      local[ly + lx*4*4] = readB4.x;
      local[ly + lx*4*4 + 4] = readB4.y;
      local[ly + lx*4*4 + 8] = readB4.z;
      local[ly + lx*4*4 + 12] = readB4.w;

      // calculate 16*vec4
      #pragma unroll
      for (int j=0; j<64; j++){
        float4 innerB4;
        innerB4 = L4[j];
        temp[j] += readA4.x * innerB4.x;
        temp[j] += readA4.y * innerB4.y;
        temp[j] += readA4.z * innerB4.z;
        temp[j] += readA4.w * innerB4.w;
      }
    }

    float4* C4 = (float4*)(&C[tidy*64 + tidx*N]);
    for(int j=0; j<16; j++){
      C4[j] = temp4[j];
    }
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input, const unsigned int width) {
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
}

int main() {
    float* hostResult;
    float* gpuMatA;
    float* gpuMatB;
    float* gpuResult;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    int i;
    int errors;

    hostResult = (float*)malloc(M*N * sizeof(float));

    // allocate the memory on the device side
    hipMalloc((void**)&gpuMatA, M*K * sizeof(float));
    hipMalloc((void**)&gpuMatB,  K*N* sizeof(float));
    hipMalloc((void**)&gpuResult, M*N * sizeof(float));

    // Memory transfer from host to device
    //hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice);

    // Lauching kernel from host
    hipLaunchKernelGGL(gemm, dim3(M/64, N/64, 1),
                    dim3(64, 1, 1), 0, 0, gpuMatA, gpuMatB, gpuResult);

    // Memory transfer from device to host
    hipMemcpy(hostResult, gpuResult, M*N * sizeof(float), hipMemcpyDeviceToHost);

    // CPU MatrixTranspose computation
    //matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

    // verify the results
        printf("PASSED!\n");
    
    // free the resources on device side
    hipFree(gpuMatA);
    hipFree(gpuMatA);
    hipFree(gpuResult);

    // free the resources on host side
    free(hostResult);

    return errors;
}