#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda.h>
#include <cuda_runtime.h>

// Type definitions for the kernel function pointers
typedef void (*NaiveKernelFunc)(const float*, const float*, float*, int, int, int, int);
typedef void (*OptimizedKernelFunc)(float*, int, int, int, int);

// Struct to hold performance metrics
struct PerformanceMetrics {
    float executionTime;  // in milliseconds
    float gflops;         // gigaflops
};

// Enum to identify kernel types
enum KernelType {
    NAIVE,
    OPTIMIZED,
    VECTORIZED
};

// Function to measure performance of the CNN kernels - returns execution time and GFLOPS
inline PerformanceMetrics measurePerformance(void* kernelFunc, KernelType kernelType,
                               float *d_input, float *d_kernel, float *d_output, 
                               int imgWidth, int imgHeight, int kernelWidth, int kernelHeight,
                               dim3 gridDim, dim3 blockDim) {
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Calculate shared memory size for kernels that need it
    int sharedMemSize = 0;
    if (kernelType == OPTIMIZED || kernelType == VECTORIZED) {
        int kernelRadiusX = kernelWidth / 2;
        int kernelRadiusY = kernelHeight / 2;
        int sharedWidth = blockDim.x + 2 * kernelRadiusX;
        int sharedHeight = blockDim.y + 2 * kernelRadiusY;
        sharedMemSize = sharedWidth * sharedHeight * sizeof(float);
    }
    
    // Warm-up run
    if (kernelType == OPTIMIZED) {
        OptimizedKernelFunc func = (OptimizedKernelFunc)kernelFunc;
        func<<<gridDim, blockDim, sharedMemSize>>>(d_output, imgWidth, imgHeight, kernelWidth, kernelHeight);
    } else {
        // Both NAIVE and VECTORIZED use the same function signature
        NaiveKernelFunc func = (NaiveKernelFunc)kernelFunc;
        func<<<gridDim, blockDim, (kernelType == VECTORIZED) ? sharedMemSize : 0>>>(
            d_input, d_kernel, d_output, imgWidth, imgHeight, kernelWidth, kernelHeight);
    }
    
    // Synchronize and start timing
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    // Number of iterations for averaging
    const int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        if (kernelType == OPTIMIZED) {
            OptimizedKernelFunc func = (OptimizedKernelFunc)kernelFunc;
            func<<<gridDim, blockDim, sharedMemSize>>>(d_output, imgWidth, imgHeight, kernelWidth, kernelHeight);
        } else {
            NaiveKernelFunc func = (NaiveKernelFunc)kernelFunc;
            func<<<gridDim, blockDim, (kernelType == VECTORIZED) ? sharedMemSize : 0>>>(
                d_input, d_kernel, d_output, imgWidth, imgHeight, kernelWidth, kernelHeight);
        }
    }
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate average execution time per iteration
    float avgExecutionTime = milliseconds / iterations;
    
    // Calculate GFLOPS
    // For convolution: each output pixel requires 2*kernelWidth*kernelHeight operations (multiply-add)
    int outputWidth = imgWidth - kernelWidth + 1;
    int outputHeight = imgHeight - kernelHeight + 1;
    double totalOperations = (double)outputWidth * outputHeight * 2 * kernelWidth * kernelHeight;
    float timeInSeconds = avgExecutionTime / 1000.0f;
    float gflops = (totalOperations / timeInSeconds) / 1e9;
    
    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return {avgExecutionTime, gflops};
}

// prefetch helper for ptx
__device__ inline void prefetch_global(const void *ptr) {
    asm volatile("prefetch.global.L1 [%0];" :: "l"(ptr));
}

// warp level reduction using shfl down sync
__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

#endif // UTILS_CUH