#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "../include/cnn_launcher.cuh"
#include <algorithm> 


int main(int argc, char** argv) {
    // Default dimensions
    int dimX = 1024;
    int dimY = 1024;
    int dimK = 3;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            dimX = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
            dimY = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            dimK = atoi(argv[i + 1]);
            i++;
        }
    }
    
    printf("Running 2D Convolution with image size: %d x %d, mask size: %d x %d\n", 
           dimX, dimY, dimK, dimK);
    
    // Allocate host memory
    size_t img_size = dimX * dimY * sizeof(float);
    size_t mask_size = dimK * dimK * sizeof(float);
    
    float *h_input = (float*)malloc(img_size);
    float *h_output_cpu = (float*)malloc(img_size);
    float *h_output_naive = (float*)malloc(img_size);
    float *h_output_optimized = (float*)malloc(img_size);
    float *h_output_vectorized = (float*)malloc(img_size);
    float *h_mask = (float*)malloc(mask_size);
    
    if (!h_input || !h_output_cpu || !h_output_naive || 
        !h_output_optimized || !h_output_vectorized || !h_mask) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        return -1;
    }
    
    // Initialize random number generator
    srand(time(NULL));
    
    // Generate random input data (0-15)
    for (int i = 0; i < dimX * dimY; i++) {
        h_input[i] = static_cast<float>(rand() % 16);
    }
    
    for (int i = 0; i < dimK * dimK; i++) {
        h_mask[i] = static_cast<float>(rand() % 16);
    }
    
    // Run CPU convolution
    printf("\nRunning CPU implementation...\n");
    PerformanceMetrics cpu_metrics = cnn_cpu(h_input, h_output_cpu, h_mask, dimX, dimY, dimK);
    
    // Run naive GPU convolution
    printf("\nRunning naive GPU implementation...\n");
    PerformanceMetrics naive_metrics = cnn_naive(h_input, h_output_naive, h_mask, dimX, dimY, dimK);
    
    // Run optimized GPU convolution
    printf("\nRunning optimized GPU implementation...\n");
    PerformanceMetrics optimized_metrics = cnn_optimized(h_input, h_output_optimized, h_mask, dimX, dimY, dimK);
    
    // Run vectorized GPU convolution
    printf("\nRunning vectorized GPU implementation...\n");
    PerformanceMetrics vectorized_metrics = cnn_vectorized(h_input, h_output_vectorized, h_mask, dimX, dimY, dimK);
    
    // Compare a few random pixels between CPU and GPU results
    int errors_naive = 0;
    int errors_optimized = 0;
    int errors_vectorized = 0;
    
    printf("\nVerifying results (random samples):\n");
    for (int i = 0; i < 10; i++) {
        int idx = rand() % (dimX * dimY);
        if (fabs(h_output_cpu[idx] - h_output_naive[idx]) > 1e-5) errors_naive++;
        if (fabs(h_output_cpu[idx] - h_output_optimized[idx]) > 1e-5) errors_optimized++;
        if (fabs(h_output_cpu[idx] - h_output_vectorized[idx]) > 1e-5) errors_vectorized++;
        
        printf("Pixel %d: CPU=%.6f, Naive=%.6f, Optimized=%.6f, Vectorized=%.6f\n",
               idx, h_output_cpu[idx], h_output_naive[idx], 
               h_output_optimized[idx], h_output_vectorized[idx]);
    }
    
    printf("\nVerification Results:\n");
    printf("Naive GPU implementation: %s\n", errors_naive == 0 ? "PASSED" : "FAILED");
    printf("Optimized GPU implementation: %s\n", errors_optimized == 0 ? "PASSED" : "FAILED");
    printf("Vectorized GPU implementation: %s\n", errors_vectorized == 0 ? "PASSED" : "FAILED");
    
    // Print performance summary
    printf("\nPerformance Summary:\n");
    printf("CPU: %f ms, %f GFLOPS\n", cpu_metrics.executionTime, cpu_metrics.gflops);
    printf("Naive GPU: %f ms, %f GFLOPS, Speedup: %.2fx\n", 
           naive_metrics.executionTime, naive_metrics.gflops,
           cpu_metrics.executionTime / naive_metrics.executionTime);
    printf("Optimized GPU: %f ms, %f GFLOPS, Speedup: %.2fx\n", 
           optimized_metrics.executionTime, optimized_metrics.gflops,
           cpu_metrics.executionTime / optimized_metrics.executionTime);
    printf("Vectorized GPU: %f ms, %f GFLOPS, Speedup: %.2fx\n", 
           vectorized_metrics.executionTime, vectorized_metrics.gflops,
           cpu_metrics.executionTime / vectorized_metrics.executionTime);
    
    // Free host memory
    free(h_input);
    free(h_output_cpu);
    free(h_output_naive);
    free(h_output_optimized);
    free(h_output_vectorized);
    free(h_mask);
    
    return 0;
}