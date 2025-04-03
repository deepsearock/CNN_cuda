#ifndef CNN_LAUNCHER_CUH
#define CNN_LAUNCHER_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cnn.cuh"
#include "utils.cuh"

// Declare the texture reference for optimized kernel (declaration only)
extern texture<float, 2, cudaReadModeElementType> texRef;

// Function prototypes

// CPU performance measurement function
PerformanceMetrics cnn_cpu(float *h_input, float *h_output, float *h_mask, 
                           int dimX, int dimY, int dimK);

// Function to launch the naive convolution kernel
PerformanceMetrics cnn_naive(float *h_input, float *h_output, float *h_mask, 
                             int dimX, int dimY, int dimK);

// Function to launch the optimized convolution kernel using texture memory
PerformanceMetrics cnn_optimized(float *h_input, float *h_output, float *h_mask, 
                                 int dimX, int dimY, int dimK);

// Function to launch the vectorized convolution kernel
PerformanceMetrics cnn_vectorized(float *h_input, float *h_output, float *h_mask, 
                                  int dimX, int dimY, int dimK);

#endif // CNN_LAUNCHE