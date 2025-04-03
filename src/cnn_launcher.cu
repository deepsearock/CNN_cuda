#include "../include/cnn_launcher.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Define the texture reference for optimized kernel
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// CPU performance measurement function
PerformanceMetrics cnn_cpu(float *h_input, float *h_output, float *h_mask, 
                           int dimX, int dimY, int dimK) {
    PerformanceMetrics metrics;
    
    // Calculate total number of operations (multiply-adds)
    long long totalOps = static_cast<long long>(dimX) * dimY * (2 * dimK * dimK - 1);
    
    // Use CUDA events for consistent timing with GPU functions
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Run CPU convolution
    cpuConvolution2D(h_input, h_mask, h_output, dimX, dimY, dimK, dimK);
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate performance metrics
    metrics.executionTime = milliseconds;
    metrics.gflops = (totalOps / (milliseconds * 1.0e6));
    
    printf("CPU Convolution Performance: %f ms, %f GFLOPS\n", 
           metrics.executionTime, metrics.gflops);
    
    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return metrics;
}

// Function to launch the naive convolution kernel
PerformanceMetrics cnn_naive(float *h_input, float *h_output, float *h_mask, 
                             int dimX, int dimY, int dimK) {
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
    PerformanceMetrics metrics = measurePerformance((void*)naiveConvolution2D, KernelType::NAIVE,
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
PerformanceMetrics cnn_optimized(float *h_input, float *h_output, float *h_mask, 
                                 int dimX, int dimY, int dimK) {
    float *d_output, *d_mask;
    size_t img_size = dimX * dimY * sizeof(float);
    size_t mask_size = dimK * dimK * sizeof(float);
    
    // Allocate device memory for output and mask
    cudaMalloc((void**)&d_output, img_size);
    cudaMalloc((void**)&d_mask, mask_size);
    
    // Copy mask data to device
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
    PerformanceMetrics metrics = measurePerformance((void*)optimizedConvolution2D, KernelType::OPTIMIZED,
                                                      nullptr, d_mask, d_output, 
                                                      dimX, dimY, dimK, dimK,
                                                      gridDim, blockDim);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    // Unbind texture and free CUDA array
    cudaUnbindTexture(texRef);
    cudaFreeArray(cuArray);
    
    // Free device memory
    cudaFree(d_output);
    cudaFree(d_mask);
    
    return metrics;
}

// Function to launch the vectorized convolution kernel
PerformanceMetrics cnn_vectorized(float *h_input, float *h_output, float *h_mask, 
                                  int dimX, int dimY, int dimK) {
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
    
    // Measure performance using the vectorized kernel
    PerformanceMetrics metrics = measurePerformance((void*)vectorizedConvolution2D, KernelType::VECTORIZED,
                                                      d_input, d_mask, d_output, 
                                                      dimX, dimY, dimK, dimK,
                                                      gridDim, blockDim);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
    
    return metrics;
}
