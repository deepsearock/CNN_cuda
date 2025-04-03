#ifndef CNN_LAUNCHER_CUH
#define CNN_LAUNCHER_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cnn.cuh"
#include "utils.cuh"

// Declare the texture reference (declaration only)
extern texture<float, 2, cudaReadModeElementType> texRef;

// Function prototypes
PerformanceMetrics cnn_cpu(float *h_input, float *h_output, float *h_mask,
                           int dimX, int dimY, int dimK);

PerformanceMetrics cnn_naive(float *h_input, float *h_output, float *h_mask,
                             int dimX, int dimY, int dimK);

PerformanceMetrics cnn_optimized(float *h_input, float *h_output, float *h_mask,
                                 int dimX, int dimY, int dimK);

PerformanceMetrics cnn_vectorized(float *h_input, float *h_output, float *h_mask,
                                  int dimX, int dimY, int dimK);

#endif // CNN_LAUNCHER_CUH
