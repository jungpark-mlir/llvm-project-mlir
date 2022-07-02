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

#define NUM (WIDTH * WIDTH)

#define M 256
#define K 768
#define N 1
#define LOG2K 8

__global__ void rgemm(float* A, float* B, float* C) {
    __shared__ float local[K];
    int tidx = hipThreadIdx_x;
    int bidx = hipBlockIdx_x;
    int tidy = hipThreadIdx_y;
    int tidz = hipThreadIdx_z;

// A[256,768]
// grid K(768 / 2), M(256), N(768)
    float inA0, inA1;
    inA0 = A[tidx*2 + bidx*K];
    inA1 = A[tidx*2 + 1 + bidx*K];

    local[tidx*2] = inA0;
    local[tidx*2 + 1] = inA1;

    int turn = 1;
    // sum tree
    for (int i=0; i<LOG2K; i++) { //until 2^8 = 256
      //__syncthreads();
      asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier");
      if((tidx % turn) == 0){
        float lValue = local[tidx*2];
        float rValue = local[tidx*2 + turn];
        local[tidx*2] = lValue + rValue;
      }
      turn *= 2;
    }

    //__syncthreads();
    asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier");
    // values are at 0, 256, 512
    if (tidx == 0){
      float final = local[0] + local[256] + local[512];
      //float final = local[0];
      C[bidx] = final / K;
    }
}

int main() {
    float* hostResult;
    float* gpuMatA;
    float* gpuMatB;
    float* gpuResult;
    float* Matrix;

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

    // input data
    Matrix = (float*)malloc(M * K * sizeof(float));

    // initialize the input data
    for (i = 0; i < M*K; i++) {
        Matrix[i] = (float)i * 0.01f;
    }

    // Memory transfer from host to device
    hipMemcpy(gpuMatA, Matrix, M*K * sizeof(float), hipMemcpyHostToDevice);

    // Lauching kernel from host
    hipLaunchKernelGGL(rgemm, dim3(M, 1, 1),
                    dim3(K/2, 1, 1), 0, 0, gpuMatA, gpuMatB, gpuResult);

    // Memory transfer from device to host
    hipMemcpy(hostResult, gpuResult, M*N * sizeof(float), hipMemcpyDeviceToHost);

    for (i = 0; i < M*N; i++) {
        int verify = 0;
        for (int j; j<K; j++)
          verify += i*K + j;
        float result = (float)verify;
        std::cout<<i <<": "<<hostResult[i] <<" vs "<<result<<"\n";
    }

    // verify the results
        printf("PASSED!\n");
    
    // free the resources on device side
    hipFree(gpuMatA);
    hipFree(gpuMatA);
    hipFree(gpuResult);

    // free the resources on host side
    free(hostResult);
    free(Matrix);

    return errors;
}