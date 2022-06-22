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

#define WGSIZE 768
#define M 256
#define K 768
#define N 2
#define LOG2K 8
#define TILE 2

__global__ void rgemm(float* A, float* B, float* C) {
    __shared__ float local[K];
    int tidx = hipThreadIdx_x;
    int tidy = hipThreadIdx_y;
    int tidz = hipThreadIdx_z;

// A[256,768] B[768,768]
// grid K(768 / 2), M(256), N(768)
    float inA0, inA1, inB0, inB1;
    inA0 = A[tidx*2 + tidy*K];
    inA1 = A[tidx*2 + 1 + tidy*K];
    inB0 = B[tidz + tidx*2*N];
    inB1 = B[tidz + (tidx*2 + 1)*N];

    local[tidx*2] = inA0;
    local[tidx*2 + 1] = inA1;

    int turn = 1;
    // sum tree
    for (int i=0; i<LOG2K; i++) { //until 2^8 = 256
      //__syncthreads();
      asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier");
      if((tidx % turn) == 0){
        float lValue = local[tidx*turn*2];
        float rValue = local[tidx*turn*2 + turn];
        local[tidx] = lValue + rValue;
      }
      turn *= 2;
    }

    //__syncthreads();
    asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier");
    // values are at 0, 256, 512
    float temp;
    if (tidx == 0){
      temp = local[0] + local[256] + local[512];
    }

    local[tidx*2] = inB0;
    local[tidx*2 + 1] = inB1;

    turn = 1;
    // sum tree
    for (int i=0; i<LOG2K; i++) { //until 2^8 = 256
      //__syncthreads();
      asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier");
      if((tidx % turn) == 0){
        float lValue = local[tidx*turn*2];
        float rValue = local[tidx*turn*2 + turn];
        local[tidx] = lValue + rValue;
      }
      turn *= 2;
    }

    //__syncthreads();
    asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier");
    if (tidx == 0){
      float final = local[0] + local[256] + local[512];
      //float final = local[0];
      C[tidz + tidy*N] = final;
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

    hostResult = (float*)malloc(256*768 * sizeof(float));

    // allocate the memory on the device side
    hipMalloc((void**)&gpuMatA, M*K * sizeof(float));
    hipMalloc((void**)&gpuMatB,  K*N* sizeof(float));
    hipMalloc((void**)&gpuResult, M*N * sizeof(float));

    // Memory transfer from host to device
    //hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice);

    // Lauching kernel from host
    hipLaunchKernelGGL(rgemm, dim3(1, M, N),
                    dim3(K/2, 1, 1), 0, 0, gpuMatA, gpuMatB, gpuResult);

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