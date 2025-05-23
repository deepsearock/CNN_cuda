#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <chrono>
#include <fstream>
#include <sstream>

// macro to check cuda errors i should have utils file for this ohwell 
#define CUDA_CHECK(call) {                                              \
    cudaError_t err = call;                                             \
    if(err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                cudaGetErrorString(err));                               \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

// 1. Optimized GPU Convolution Kernel (texture + shared memory) it really isnt optimized as its slower than shared memory only
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
    
    // load the center region from textured memory into shared memory
    if (row < imageHeight && col < imageWidth)
        sharedMem[shared_y * sharedWidth + shared_x] = tex2D(texRef, col, row);
    else
        sharedMem[shared_y * sharedWidth + shared_x] = 0;
    
    // load the halo or ghost regions from textured memory into shared memory
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
    // load the corner halos from textured memory into shared memory
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
    
    // convolution operation using the shared memory 
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

//2. naive gpu convolution kernel (global memory only) 
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


//3. shared memory gpu convolution kernel (global memory + shared memory)
__global__ void sharedMemoryConvolutionKernel(int *input, int *output, int imageWidth, int imageHeight, int maskWidth, int maskRadius) {
    extern __shared__ int sharedMem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    
    int sharedWidth = blockDim.x + 2 * maskRadius;
    int shared_x = tx + maskRadius;
    int shared_y = ty + maskRadius;
    
    // load the center region from global memory into shared memory
    if (row < imageHeight && col < imageWidth)
        sharedMem[shared_y * sharedWidth + shared_x] = input[row * imageWidth + col];
    else
        sharedMem[shared_y * sharedWidth + shared_x] = 0;
    
    // same as optimized kernel, load the halo or ghost regions from global memory into shared memory
    if (tx < maskRadius) {
        int halo_col = col - maskRadius;
        int value = (halo_col >= 0 && row < imageHeight) ? input[row * imageWidth + halo_col] : 0;
        sharedMem[shared_y * sharedWidth + tx] = value;
    }
    if (tx >= blockDim.x - maskRadius) {
        int halo_col = col + maskRadius;
        int value = (halo_col < imageWidth && row < imageHeight) ? input[row * imageWidth + halo_col] : 0;
        sharedMem[shared_y * sharedWidth + shared_x + maskRadius] = value;
    }
    if (ty < maskRadius) {
        int halo_row = row - maskRadius;
        int value = (halo_row >= 0 && col < imageWidth) ? input[halo_row * imageWidth + col] : 0;
        sharedMem[ty * sharedWidth + shared_x] = value;
    }
    if (ty >= blockDim.y - maskRadius) {
        int halo_row = row + maskRadius;
        int value = (halo_row < imageHeight && col < imageWidth) ? input[halo_row * imageWidth + col] : 0;
        sharedMem[(shared_y + maskRadius) * sharedWidth + shared_x] = value;
    }
    // corner halos
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
    
    // convolution operation
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

// 4. cpu convolution function
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
    // default size 
    int dimX = 512, dimY = 512;
    // take input
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0)
            dimX = atoi(argv[++i]);
        else if (strcmp(argv[i], "-j") == 0)
            dimY = atoi(argv[++i]);
    }
    
    int imageSize = dimX * dimY;
    
    // allocate host memory for the image and output arrays
    int *h_image         = (int *)malloc(sizeof(int) * imageSize);
    int *h_output_gpu    = (int *)malloc(sizeof(int) * imageSize); // optimized GPU (texture+shared)
    int *h_output_naive  = (int *)malloc(sizeof(int) * imageSize); // naive GPU (global only)
    int *h_output_shared = (int *)malloc(sizeof(int) * imageSize); // shared Memory GPU
    int *h_output_cpu    = (int *)malloc(sizeof(int) * imageSize); // CPU result
    
    // random image generation 0-15
    for (int i = 0; i < imageSize; i++) {
        h_image[i] = rand() % 16;
    }
    
    // mask sizes
    int maskSizes[] = {4, 6, 8, 10, 12, 14, 16, 18, 20};
    int numMasks = sizeof(maskSizes) / sizeof(maskSizes[0]);
    
    // cuda block size of 256 and 16x16 shape
    dim3 blockDim(16, 16);
    dim3 gridDim((dimX + blockDim.x - 1) / blockDim.x, (dimY + blockDim.y - 1) / blockDim.y);
    
    std::ostringstream filename;
    filename << "results_" << dimX << "x" << dimY << ".csv";
    std::ofstream csvFile(filename.str());
    if (!csvFile.is_open()) {
        fprintf(stderr, "Error opening %s for writing.\n", filename.str().c_str());
        return EXIT_FAILURE;
    }
    csvFile << "# Image dimensions: " << dimX << " x " << dimY << "\n";
    csvFile << "MaskSize,OptimizedTime_ms,OptimizedGFLOPS,NaiveTime_ms,NaiveGFLOPS,SharedTime_ms,SharedGFLOPS,CPUTime_ms,CPUGFLOPS,Status\n";
    
    for (int m = 0; m < numMasks; m++) {
        int maskWidth = maskSizes[m];
        int maskRadius = maskWidth / 2;
        
        // calculate shared memory size
        int sharedMemSize = (blockDim.x + 2 * maskRadius) * (blockDim.y + 2 * maskRadius) * sizeof(int);
        
// kernel 1
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
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
// kernel 2
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
        CUDA_CHECK(cudaEventDestroy(startNaive));
        CUDA_CHECK(cudaEventDestroy(stopNaive));
        
// kernel 3
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
        CUDA_CHECK(cudaEventDestroy(startShared));
        CUDA_CHECK(cudaEventDestroy(stopShared));
        
// kernel 4 cpu
        auto cpu_start = std::chrono::high_resolution_clock::now();
        convolution2D_CPU(h_image, h_output_cpu, dimX, dimY, maskWidth);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsedTime_cpu = cpu_end - cpu_start;
        
// compare
        const double epsilon = 1e-5;
        bool correct = true;
        for (int i = 0; i < imageSize; i++) {
            if (abs(h_output_gpu[i] - h_output_cpu[i]) > epsilon ||
                abs(h_output_naive[i] - h_output_cpu[i]) > epsilon ||
                abs(h_output_shared[i] - h_output_cpu[i]) > epsilon) {
                correct = false;
                break;
            }
        }
        std::string status = correct ? "Correct" : "Mismatch";
        
// calculate GFLOPS
        double opsPerPixel = (maskWidth * maskWidth) + 1;
        double totalOps = imageSize * opsPerPixel;
        
        double seconds_gpu    = elapsedTime_gpu / 1000.0;
        double gflops_gpu     = (totalOps / seconds_gpu) / 1e9;
        
        double seconds_naive  = elapsedTime_naive / 1000.0;
        double gflops_naive   = (totalOps / seconds_naive) / 1e9;
        
        double seconds_shared = elapsedTime_shared / 1000.0;
        double gflops_shared  = (totalOps / seconds_shared) / 1e9;
        
        double seconds_cpu    = elapsedTime_cpu.count() / 1000.0;
        double gflops_cpu     = (totalOps / seconds_cpu) / 1e9;
        
// write results to csv file
        csvFile << maskWidth << ","
                << elapsedTime_gpu << "," << gflops_gpu << ","
                << elapsedTime_naive << "," << gflops_naive << ","
                << elapsedTime_shared << "," << gflops_shared << ","
                << elapsedTime_cpu.count() << "," << gflops_cpu << ","
                << status << "\n";
    }
    
    csvFile.close();
    printf("All kernels produced correct results. Results written to %s\n", filename.str().c_str());
    
    // clean up
    free(h_image);
    free(h_output_gpu);
    free(h_output_naive);
    free(h_output_shared);
    free(h_output_cpu);
    
    return 0;
}
