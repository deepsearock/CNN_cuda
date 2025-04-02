#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include "cnn_naive.cuh"  // Include the header file with the naive kernel
#include "utils.cuh"  // Include utils for performance measurement

// Function to launch the convolution kernel
PerformanceMetrics cnn_naive(float *h_input, float *h_output, float *h_mask, int dimX, int dimY, int dimK) {
    float *d_input, *d_output, *d_mask;
    size_t img_size = dimX * dimY * sizeof(float);
    size_t mask_size = dimK * dimK * sizeof(float);
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, img_size);
    cudaMalloc((void**)&d_output, img_size);
    cudaMalloc((void**)&d_mask, mask_size);
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, mask_size, cudaMemcpyHostToDevice);
    
    // Set grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((dimX + blockDim.x - 1) / blockDim.x, 
                 (dimY + blockDim.y - 1) / blockDim.y);
    
    // Measure performance using the naive kernel
    PerformanceMetrics metrics = measurePerformance((void*)naiveConvolution2D, false,
                                                   d_input, d_mask, d_output, 
                                                   dimX, dimY, dimK, dimK,
                                                   gridDim, blockDim);
    
    printf("Naive Convolution Performance: %f ms, %f GFLOPS\n", 
           metrics.executionTime, metrics.gflops);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
    
    return metrics;
}

// Function to launch the optimized convolution kernel using texture memory
PerformanceMetrics cnn_optimized(float *h_input, float *h_output, float *h_mask, int dimX, int dimY, int dimK) {
    float *d_input, *d_output, *d_mask;
    size_t img_size = dimX * dimY * sizeof(float);
    size_t mask_size = dimK * dimK * sizeof(float);
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, img_size);
    cudaMalloc((void**)&d_output, img_size);
    cudaMalloc((void**)&d_mask, mask_size);
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, mask_size, cudaMemcpyHostToDevice);
    
    // Set up texture reference for input image
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, dimX, dimY);
    cudaMemcpyToArray(cuArray, 0, 0, h_input, img_size, cudaMemcpyHostToDevice);
    
    // Bind texture reference to the CUDA array
    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.filterMode = cudaFilterModePoint;
    texRef.normalized = false;
    cudaBindTextureToArray(texRef, cuArray, channelDesc);
    
    // Set grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((dimX + blockDim.x - 1) / blockDim.x, 
                 (dimY + blockDim.y - 1) / blockDim.y);
    
    // Measure performance using the optimized kernel
    PerformanceMetrics metrics = measurePerformance((void*)optimizedConvolution2D, true,
                                                   d_input, d_mask, d_output, 
                                                   dimX, dimY, dimK, dimK,
                                                   gridDim, blockDim);
    
    printf("Optimized Convolution Performance: %f ms, %f GFLOPS\n", 
           metrics.executionTime, metrics.gflops);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    // Unbind texture
    cudaUnbindTexture(texRef);
    
    // Free device memory
    cudaFreeArray(cuArray);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
    
    return metrics;
}