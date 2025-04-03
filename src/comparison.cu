#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <chrono>

// Macro for error checking
#define CUDA_CHECK(call) {                                              \
    cudaError_t err = call;                                             \
    if(err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                cudaGetErrorString(err));                               \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

//------------------------------------------------------
// Existing Kernel: Optimized GPU Convolution using Texture & Shared Memory
//------------------------------------------------------
texture<int, cudaTextureType2D, cudaReadModeElementType> texRef;

__global__ void convolution2DKernel(int *output, int imageWidth, int imageHeight, int maskWidth, int maskRadius) {
    extern __shared__ int sharedMem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    
    int sharedWidth = blockDim.x + 2 * maskRadius;
    int shared_x = tx + maskRadius;
    int shared_y = ty + maskRadius;
    
    // Load central data from texture memory
    if (row < imageHeight && col < imageWidth)
        sharedMem[shared_y * sharedWidth + shared_x] = tex2D(texRef, col, row);
    else
        sharedMem[shared_y * sharedWidth + shared_x] = 0;
    
    // Load halo regions (left, right, top, bottom, and corners)
    if (tx < maskRadius) {
        int halo_col = col - maskRadius;
        int value = (halo_col >= 0 && row < imageHeight) ? tex2D(texRef, halo_col, row) : 0;
        sharedMem[shared_y * sharedWidth + tx] = value;
    }
    if (tx >= blockDim.x - maskRadius) {
        int halo_col = col + maskRadius;
        int value = (halo_col < imageWidth && row < imageHeight) ? tex2D(texRef, halo_col, row) : 0;
        sharedMem[shared_y * sharedWidth + shared_x + maskRadius] = value;
    }
    if (ty < maskRadius) {
        int halo_row = row - maskRadius;
        int value = (halo_row >= 0 && col < imageWidth) ? tex2D(texRef, col, halo_row) : 0;
        sharedMem[ty * sharedWidth + shared_x] = value;
    }
    if (ty >= blockDim.y - maskRadius) {
        int halo_row = row + maskRadius;
        int value = (halo_row < imageHeight && col < imageWidth) ? tex2D(texRef, col, halo_row) : 0;
        sharedMem[(shared_y + maskRadius) * sharedWidth + shared_x] = value;
    }
    // Corner halos:
    if (tx < maskRadius && ty < maskRadius) {
        int halo_col = col - maskRadius;
        int halo_row = row - maskRadius;
        int value = (halo_col >= 0 && halo_row >= 0) ? tex2D(texRef, halo_col, halo_row) : 0;
        sharedMem[ty * sharedWidth + tx] = value;
    }
    if (tx >= blockDim.x - maskRadius && ty < maskRadius) {
        int halo_col = col + maskRadius;
        int halo_row = row - maskRadius;
        int value = (halo_col < imageWidth && halo_row >= 0) ? tex2D(texRef, halo_col, halo_row) : 0;
        sharedMem[ty * sharedWidth + shared_x + maskRadius] = value;
    }
    if (tx < maskRadius && ty >= blockDim.y - maskRadius) {
        int halo_col = col - maskRadius;
        int halo_row = row + maskRadius;
        int value = (halo_col >= 0 && halo_row < imageHeight) ? tex2D(texRef, halo_col, halo_row) : 0;
        sharedMem[(shared_y + maskRadius) * sharedWidth + tx] = value;
    }
    if (tx >= blockDim.x - maskRadius && ty >= blockDim.y - maskRadius) {
        int halo_col = col + maskRadius;
        int halo_row = row + maskRadius;
        int value = (halo_col < imageWidth && halo_row < imageHeight) ? tex2D(texRef, halo_col, halo_row) : 0;
        sharedMem[(shared_y + maskRadius) * sharedWidth + shared_x + maskRadius] = value;
    }
    
    __syncthreads();
    
    // Convolution operation using shared memory tile
    if (row < imageHeight && col < imageWidth) {
        int output_value = 0;
        for (int i = -maskRadius; i <= maskRadius; i++) {
            for (int j = -maskRadius; j <= maskRadius; j++) {
                int image_value = sharedMem[(shared_y + i) * sharedWidth + (shared_x + j)];
                output_value += image_value;
            }
        }
        output_value /= ((2 * maskRadius + 1) * (2 * maskRadius + 1));
        output[row * imageWidth + col] = output_value;
    }
}

//------------------------------------------------------
// Existing Kernel: Naive GPU Convolution (Global Memory Only)
//------------------------------------------------------
__global__ void naiveConvolutionKernel(int *input, int *output, int imageWidth, int imageHeight, int maskWidth, int maskRadius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < imageHeight && col < imageWidth) {
        int sum = 0;
        for (int i = -maskRadius; i <= maskRadius; i++) {
            for (int j = -maskRadius; j <= maskRadius; j++) {
                int r = row + i;
                int c = col + j;
                int value = 0;
                if (r >= 0 && r < imageHeight && c >= 0 && c < imageWidth)
                    value = input[r * imageWidth + c];
                sum += value;
            }
        }
        output[row * imageWidth + col] = sum / ((2 * maskRadius + 1) * (2 * maskRadius + 1));
    }
}

//------------------------------------------------------
// NEW Kernel: Shared Memory Convolution (Global Memory Input + Shared Memory)
//------------------------------------------------------
__global__ void sharedMemoryConvolutionKernel(int *input, int *output, int imageWidth, int imageHeight, int maskWidth, int maskRadius) {
    extern __shared__ int sharedMem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    
    int sharedWidth = blockDim.x + 2 * maskRadius;
    int shared_x = tx + maskRadius;
    int shared_y = ty + maskRadius;
    
    // Load central region from global memory into shared memory
    if (row < imageHeight && col < imageWidth)
        sharedMem[shared_y * sharedWidth + shared_x] = input[row * imageWidth + col];
    else
        sharedMem[shared_y * sharedWidth + shared_x] = 0;
    
    // Load halo regions from global memory.
    // Left halo:
    if (tx < maskRadius) {
        int halo_col = col - maskRadius;
        int value = (halo_col >= 0 && row < imageHeight) ? input[row * imageWidth + halo_col] : 0;
        sharedMem[shared_y * sharedWidth + tx] = value;
    }
    // Right halo:
    if (tx >= blockDim.x - maskRadius) {
        int halo_col = col + maskRadius;
        int value = (halo_col < imageWidth && row < imageHeight) ? input[row * imageWidth + halo_col] : 0;
        sharedMem[shared_y * sharedWidth + shared_x + maskRadius] = value;
    }
    // Top halo:
    if (ty < maskRadius) {
        int halo_row = row - maskRadius;
        int value = (halo_row >= 0 && col < imageWidth) ? input[halo_row * imageWidth + col] : 0;
        sharedMem[ty * sharedWidth + shared_x] = value;
    }
    // Bottom halo:
    if (ty >= blockDim.y - maskRadius) {
        int halo_row = row + maskRadius;
        int value = (halo_row < imageHeight && col < imageWidth) ? input[halo_row * imageWidth + col] : 0;
        sharedMem[(shared_y + maskRadius) * sharedWidth + shared_x] = value;
    }
    // Corner halos:
    if (tx < maskRadius && ty < maskRadius) {
        int halo_col = col - maskRadius;
        int halo_row = row - maskRadius;
        int value = (halo_col >= 0 && halo_row >= 0) ? input[halo_row * imageWidth + halo_col] : 0;
        sharedMem[ty * sharedWidth + tx] = value;
    }
    if (tx >= blockDim.x - maskRadius && ty < maskRadius) {
        int halo_col = col + maskRadius;
        int halo_row = row - maskRadius;
        int value = (halo_col < imageWidth && halo_row >= 0) ? input[halo_row * imageWidth + halo_col] : 0;
        sharedMem[ty * sharedWidth + shared_x + maskRadius] = value;
    }
    if (tx < maskRadius && ty >= blockDim.y - maskRadius) {
        int halo_col = col - maskRadius;
        int halo_row = row + maskRadius;
        int value = (halo_col >= 0 && halo_row < imageHeight) ? input[halo_row * imageWidth + halo_col] : 0;
        sharedMem[(shared_y + maskRadius) * sharedWidth + tx] = value;
    }
    if (tx >= blockDim.x - maskRadius && ty >= blockDim.y - maskRadius) {
        int halo_col = col + maskRadius;
        int halo_row = row + maskRadius;
        int value = (halo_col < imageWidth && halo_row < imageHeight) ? input[halo_row * imageWidth + halo_col] : 0;
        sharedMem[(shared_y + maskRadius) * sharedWidth + shared_x + maskRadius] = value;
    }
    
    __syncthreads();
    
    // Convolution operation using the shared memory tile
    if (row < imageHeight && col < imageWidth) {
        int sum = 0;
        for (int i = -maskRadius; i <= maskRadius; i++) {
            for (int j = -maskRadius; j <= maskRadius; j++) {
                sum += sharedMem[(shared_y + i) * sharedWidth + (shared_x + j)];
            }
        }
        output[row * imageWidth + col] = sum / ((2 * maskRadius + 1) * (2 * maskRadius + 1));
    }
}

//------------------------------------------------------
// CPU Convolution Implementation
//------------------------------------------------------
void convolution2D_CPU(const int *h_image, int *h_output, int imageWidth, int imageHeight, int maskWidth) {
    int maskRadius = maskWidth / 2;
    for (int row = 0; row < imageHeight; row++) {
        for (int col = 0; col < imageWidth; col++) {
            int sum = 0;
            for (int i = -maskRadius; i <= maskRadius; i++) {
                for (int j = -maskRadius; j <= maskRadius; j++) {
                    int curRow = row + i;
                    int curCol = col + j;
                    int value = 0;
                    if (curRow >= 0 && curRow < imageHeight && curCol >= 0 && curCol < imageWidth)
                        value = h_image[curRow * imageWidth + curCol];
                    sum += value;
                }
            }
            h_output[row * imageWidth + col] = sum / ((2 * maskRadius + 1) * (2 * maskRadius + 1));
        }
    }
}

int main(int argc, char **argv) {
    // Default dimensions: 512x512 image with a 3x3 mask.
    int dimX = 512, dimY = 512, dimK = 3;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0)
            dimX = atoi(argv[++i]);
        else if (strcmp(argv[i], "-j") == 0)
            dimY = atoi(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0)
            dimK = atoi(argv[++i]);
    }
    
    int imageSize = dimX * dimY;
    
    // Allocate host memory for image and results.
    int *h_image         = (int *)malloc(sizeof(int) * imageSize);
    int *h_output_gpu      = (int *)malloc(sizeof(int) * imageSize); // optimized GPU (texture+shared)
    int *h_output_naive    = (int *)malloc(sizeof(int) * imageSize); // naive GPU (global only)
    int *h_output_shared   = (int *)malloc(sizeof(int) * imageSize); // new shared-memory GPU kernel
    int *h_output_cpu      = (int *)malloc(sizeof(int) * imageSize); // CPU result
    
    // Random image values between 0 and 15.
    for (int i = 0; i < imageSize; i++) {
        h_image[i] = rand() % 16;
    }
    
    int maskWidth = dimK;
    int maskRadius = maskWidth / 2;
    
    //------------------------------------------------------
    // Optimized GPU Convolution (Texture + Shared Memory)
    //------------------------------------------------------
    cudaArray *d_imageArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
    CUDA_CHECK(cudaMallocArray(&d_imageArray, &channelDesc, dimX, dimY));
    
    CUDA_CHECK(cudaMemcpy2DToArray(d_imageArray, 0, 0, h_image, dimX * sizeof(int),
                                   dimX * sizeof(int), dimY, cudaMemcpyHostToDevice));
    
    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.filterMode     = cudaFilterModePoint;
    texRef.normalized     = false;
    CUDA_CHECK(cudaBindTextureToArray(texRef, d_imageArray, channelDesc));
    
    int *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int) * imageSize));
    
    dim3 blockDim(16, 16);
    dim3 gridDim((dimX + blockDim.x - 1) / blockDim.x, (dimY + blockDim.y - 1) / blockDim.y);
    int sharedMemSize = (blockDim.x + 2 * maskRadius) * (blockDim.y + 2 * maskRadius) * sizeof(int);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start, 0));
    convolution2DKernel<<<gridDim, blockDim, sharedMemSize>>>(d_output, dimX, dimY, maskWidth, maskRadius);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsedTime_gpu;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_gpu, start, stop));
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, sizeof(int) * imageSize, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaUnbindTexture(texRef));
    CUDA_CHECK(cudaFreeArray(d_imageArray));
    CUDA_CHECK(cudaFree(d_output));
    
    //------------------------------------------------------
    // Naive GPU Convolution (Global Memory Only)
    //------------------------------------------------------
    int *d_input, *d_output_naive;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(int) * imageSize));
    CUDA_CHECK(cudaMalloc(&d_output_naive, sizeof(int) * imageSize));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_image, sizeof(int) * imageSize, cudaMemcpyHostToDevice));
    
    cudaEvent_t startNaive, stopNaive;
    CUDA_CHECK(cudaEventCreate(&startNaive));
    CUDA_CHECK(cudaEventCreate(&stopNaive));
    
    CUDA_CHECK(cudaEventRecord(startNaive, 0));
    naiveConvolutionKernel<<<gridDim, blockDim>>>(d_input, d_output_naive, dimX, dimY, maskWidth, maskRadius);
    CUDA_CHECK(cudaEventRecord(stopNaive, 0));
    CUDA_CHECK(cudaEventSynchronize(stopNaive));
    
    float elapsedTime_naive;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_naive, startNaive, stopNaive));
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(h_output_naive, d_output_naive, sizeof(int) * imageSize, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output_naive));
    
    //------------------------------------------------------
    // NEW: Shared Memory GPU Convolution (Global Memory + Shared Memory)
    //------------------------------------------------------
    int *d_input_shared, *d_output_shared;
    CUDA_CHECK(cudaMalloc(&d_input_shared, sizeof(int) * imageSize));
    CUDA_CHECK(cudaMalloc(&d_output_shared, sizeof(int) * imageSize));
    
    CUDA_CHECK(cudaMemcpy(d_input_shared, h_image, sizeof(int) * imageSize, cudaMemcpyHostToDevice));
    
    cudaEvent_t startShared, stopShared;
    CUDA_CHECK(cudaEventCreate(&startShared));
    CUDA_CHECK(cudaEventCreate(&stopShared));
    
    CUDA_CHECK(cudaEventRecord(startShared, 0));
    sharedMemoryConvolutionKernel<<<gridDim, blockDim, sharedMemSize>>>(d_input_shared, d_output_shared, dimX, dimY, maskWidth, maskRadius);
    CUDA_CHECK(cudaEventRecord(stopShared, 0));
    CUDA_CHECK(cudaEventSynchronize(stopShared));
    
    float elapsedTime_shared;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_shared, startShared, stopShared));
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(h_output_shared, d_output_shared, sizeof(int) * imageSize, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_input_shared));
    CUDA_CHECK(cudaFree(d_output_shared));
    
    //------------------------------------------------------
    // CPU Convolution
    //------------------------------------------------------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    convolution2D_CPU(h_image, h_output_cpu, dimX, dimY, maskWidth);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedTime_cpu = cpu_end - cpu_start;
    
    //------------------------------------------------------
    // Compare Results (Optimized, Naive, and Shared Memory GPU vs CPU)
    //------------------------------------------------------
    int errorCount = 0;
    for (int i = 0; i < imageSize; i++) {
        if (abs(h_output_gpu[i] - h_output_cpu[i]) > 1e-5) {
            errorCount++;
            if (errorCount < 10) {
                printf("Optimized GPU mismatch at index %d: GPU = %d, CPU = %d\n", i, h_output_gpu[i], h_output_cpu[i]);
            }
        }
    }
    if (errorCount == 0)
        printf("Optimized GPU and CPU results match.\n");
    else
        printf("Total mismatches (Optimized GPU vs CPU): %d\n", errorCount);
    
    errorCount = 0;
    for (int i = 0; i < imageSize; i++) {
        if (abs(h_output_naive[i] - h_output_cpu[i]) > 1e-5) {
            errorCount++;
            if (errorCount < 10) {
                printf("Naive GPU mismatch at index %d: GPU = %d, CPU = %d\n", i, h_output_naive[i], h_output_cpu[i]);
            }
        }
    }
    if (errorCount == 0)
        printf("Naive GPU and CPU results match.\n");
    else
        printf("Total mismatches (Naive GPU vs CPU): %d\n", errorCount);
    
    errorCount = 0;
    for (int i = 0; i < imageSize; i++) {
        if (abs(h_output_shared[i] - h_output_cpu[i]) > 1e-5) {
            errorCount++;
            if (errorCount < 10) {
                printf("Shared Memory GPU mismatch at index %d: GPU = %d, CPU = %d\n", i, h_output_shared[i], h_output_cpu[i]);
            }
        }
    }
    if (errorCount == 0)
        printf("Shared Memory GPU and CPU results match.\n");
    else
        printf("Total mismatches (Shared Memory GPU vs CPU): %d\n", errorCount);
    
    //------------------------------------------------------
    // Performance Calculations
    //------------------------------------------------------
    double opsPerPixel = (maskWidth * maskWidth) + 1;
    double totalOps = imageSize * opsPerPixel;
    
    double seconds_gpu   = elapsedTime_gpu / 1000.0;
    double gflops_gpu    = (totalOps / seconds_gpu) / 1e9;
    
    double seconds_naive = elapsedTime_naive / 1000.0;
    double gflops_naive  = (totalOps / seconds_naive) / 1e9;
    
    double seconds_shared = elapsedTime_shared / 1000.0;
    double gflops_shared  = (totalOps / seconds_shared) / 1e9;
    
    double seconds_cpu   = elapsedTime_cpu.count() / 1000.0;
    double gflops_cpu    = (totalOps / seconds_cpu) / 1e9;
    
    printf("Optimized GPU kernel execution time: %f ms, Performance: %f GFLOPS\n", elapsedTime_gpu, gflops_gpu);
    printf("Naive GPU kernel execution time: %f ms, Performance: %f GFLOPS\n", elapsedTime_naive, gflops_naive);
    printf("Shared Memory GPU kernel execution time: %f ms, Performance: %f GFLOPS\n", elapsedTime_shared, gflops_shared);
    printf("CPU execution time: %f ms, Performance: %f GFLOPS\n", elapsedTime_cpu.count(), gflops_cpu);
    
    // (Optional) Print a sample of each output.
    printf("Sample output (Optimized GPU):\n");
    for (int i = 0; i < 10; i++) printf("%d ", h_output_gpu[i]);
    printf("\n");
    
    printf("Sample output (Naive GPU):\n");
    for (int i = 0; i < 10; i++) printf("%d ", h_output_naive[i]);
    printf("\n");
    
    printf("Sample output (Shared Memory GPU):\n");
    for (int i = 0; i < 10; i++) printf("%d ", h_output_shared[i]);
    printf("\n");
    
    // Clean up CUDA events.
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(startNaive));
    CUDA_CHECK(cudaEventDestroy(stopNaive));
    CUDA_CHECK(cudaEventDestroy(startShared));
    CUDA_CHECK(cudaEventDestroy(stopShared));
    
    free(h_image);
    free(h_output_gpu);
    free(h_output_naive);
    free(h_output_shared);
    free(h_output_cpu);
    
    return 0;
}
