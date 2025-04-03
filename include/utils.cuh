#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Typedefs for kernel function pointers
typedef void (*NaiveKernelFunc)(const float*, const float*, float*, int, int, int, int);
typedef void (*OptimizedKernelFunc)(const float*, float*, int, int, int, int);


// Structure to hold performance metrics
struct PerformanceMetrics {
    float executionTime; // Average execution time in milliseconds
    float gflops;        // Gigaflops computed from total operations
};

// Enum to identify kernel types
enum KernelType {
    NAIVE,
    OPTIMIZED,
    VECTORIZED
};

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",             \
                    __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while(0)

// Inline function to measure the performance of CNN kernels.
// Note: For the NAIVE and VECTORIZED kernels, the expected kernel signature is:
//       void kernel(const float*, const float*, float*, int, int, int, int)
// For the OPTIMIZED kernel, the signature is:
//       void kernel(float*, int, int, int, int)
inline PerformanceMetrics measurePerformance(void* kernelFunc, KernelType kernelType,
                                               float *d_input, float *d_kernel, float *d_output, 
                                               int imgWidth, int imgHeight, int kernelWidth, int kernelHeight,
                                               dim3 gridDim, dim3 blockDim) {
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Calculate shared memory size for OPTIMIZED or VECTORIZED kernels.
    int sharedMemSize = 0;
    if (kernelType == OPTIMIZED || kernelType == VECTORIZED) {
        int kernelRadiusX = kernelWidth / 2;
        int kernelRadiusY = kernelHeight / 2;
        int sharedWidth = blockDim.x + 2 * kernelRadiusX;
        int sharedHeight = blockDim.y + 2 * kernelRadiusY;
        sharedMemSize = sharedWidth * sharedHeight * sizeof(float);
    }
    
    // Warm-up run (to avoid first-run overhead)
    if (kernelType == OPTIMIZED) {
        OptimizedKernelFunc func = (OptimizedKernelFunc)kernelFunc;
    func<<<gridDim, blockDim, sharedMemSize>>>(d_kernel, d_output, imgWidth, imgHeight, kernelWidth, kernelHeight);
    } else {
        NaiveKernelFunc func = (NaiveKernelFunc)kernelFunc;
        func<<<gridDim, blockDim, (kernelType == VECTORIZED) ? sharedMemSize : 0>>>(
            d_input, d_kernel, d_output, imgWidth, imgHeight, kernelWidth, kernelHeight);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Start timing
    CUDA_CHECK(cudaEventRecord(start));
    
    const int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        if (kernelType == OPTIMIZED) {
            OptimizedKernelFunc func = (OptimizedKernelFunc)kernelFunc;
    func<<<gridDim, blockDim, sharedMemSize>>>(d_kernel, d_output, imgWidth, imgHeight, kernelWidth, kernelHeight);
        } else {
            NaiveKernelFunc func = (NaiveKernelFunc)kernelFunc;
            func<<<gridDim, blockDim, (kernelType == VECTORIZED) ? sharedMemSize : 0>>>(
                d_input, d_kernel, d_output, imgWidth, imgHeight, kernelWidth, kernelHeight);
        }
    }
    
    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    float avgExecutionTime = milliseconds / iterations;
    
    // Calculate GFLOPS:
    // Assumes "valid" convolution: output dimensions are (imgWidth - kernelWidth + 1) x (imgHeight - kernelHeight + 1)
    int outputWidth = imgWidth - kernelWidth + 1;
    int outputHeight = imgHeight - kernelHeight + 1;
    double totalOperations = (double)outputWidth * outputHeight * 2 * kernelWidth * kernelHeight;
    float timeInSeconds = avgExecutionTime / 1000.0f;
    float gflops = (totalOperations / timeInSeconds) / 1e9;
    
    // Clean up events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    PerformanceMetrics metrics = {avgExecutionTime, gflops};
    return metrics;
}

// Device helper: prefetch data from global memory into cache
__device__ inline void prefetch_global(const void *ptr) {
    asm volatile("prefetch.global.L1 [%0];" :: "l"(ptr));
}

// Device helper: warp-level reduction using shuffle down
__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

#endif // UTILS_CUH
