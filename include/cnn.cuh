#ifndef CNN_CUH
#define CNN_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>

// Texture reference for 2D image
extern texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
// CPU implementation of 2D convolution
void cpuConvolution2D(const float* input, const float* kernel, float* output, 
                      int imgWidth, int imgHeight, int kernelWidth, int kernelHeight) {
    int kernelRadiusX = kernelWidth / 2;
    int kernelRadiusY = kernelHeight / 2;
    
    // For each pixel in the output image
    for (int y = 0; y < imgHeight; ++y) {
        for (int x = 0; x < imgWidth; ++x) {
            float sum = 0.0f;
            
            // Apply the kernel
            for (int ky = -kernelRadiusY; ky <= kernelRadiusY; ++ky) {
                for (int kx = -kernelRadiusX; kx <= kernelRadiusX; ++kx) {
                    // Handle boundary conditions with clamping
                    int imgX = min(max(x + kx, 0), imgWidth - 1);
                    int imgY = min(max(y + ky, 0), imgHeight - 1);
                    
                    // Get the corresponding kernel value
                    int kernelX = kx + kernelRadiusX;
                    int kernelY = ky + kernelRadiusY;
                    
                    sum += input[imgY * imgWidth + imgX] * 
                           kernel[kernelY * kernelWidth + kernelX];
                }
            }
            
            // Store the result
            output[y * imgWidth + x] = sum;
        }
    }
}

// Naive convolution kernel
__global__ void naiveConvolution2D(const float* input, const float* kernel, float* output, int imgWidth, int imgHeight, int kernelWidth, int kernelHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < imgWidth && y < imgHeight) {
        float sum = 0.0f;
        int kernelRadiusX = kernelWidth / 2;
        int kernelRadiusY = kernelHeight / 2;

        for (int ky = -kernelRadiusY; ky <= kernelRadiusY; ++ky) {
            for (int kx = -kernelRadiusX; kx <= kernelRadiusX; ++kx) {
                int imgX = min(max(x + kx, 0), imgWidth - 1);
                int imgY = min(max(y + ky, 0), imgHeight - 1);
                sum += input[imgY * imgWidth + imgX] * kernel[(ky + kernelRadiusY) * kernelWidth + (kx + kernelRadiusX)];
            }
        }
        output[y * imgWidth + x] = sum;
    }
}

// Optimized convolution kernel using shared memory and texture memory
__global__ void optimizedConvolution2D(float* output, int imgWidth, int imgHeight, int kernelWidth, int kernelHeight) {
    extern __shared__ float sharedMem[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    int kernelRadiusX = kernelWidth / 2;
    int kernelRadiusY = kernelHeight / 2;

    // Load data into shared memory
    int sharedWidth = blockDim.x + 2 * kernelRadiusX;
    // Removed unused variable sharedHeight

    int sharedX = threadX + kernelRadiusX;
    int sharedY = threadY + kernelRadiusY;

    if (x < imgWidth && y < imgHeight) {
        sharedMem[sharedY * sharedWidth + sharedX] = tex2D<float>(texRef, x, y);
    }

    // Load halo regions
    if (threadX < kernelRadiusX) {
        if (x >= kernelRadiusX) {
            sharedMem[sharedY * sharedWidth + threadX] = tex2D(texRef, x - kernelRadiusX, y);
        }
        if (x + blockDim.x < imgWidth) {
            sharedMem[sharedY * sharedWidth + sharedX + blockDim.x] = tex2D(texRef, x + blockDim.x, y);
        }
    }

    if (threadY < kernelRadiusY) {
        if (y >= kernelRadiusY) {
            sharedMem[threadY * sharedWidth + sharedX] = tex2D(texRef, x, y - kernelRadiusY);
        }
        if (y + blockDim.y < imgHeight) {
            sharedMem[(sharedY + blockDim.y) * sharedWidth + sharedX] = tex2D(texRef, x, y + blockDim.y);
        }
    }

    __syncthreads();

    // Perform convolution
    if (x < imgWidth && y < imgHeight) {
        float sum = 0.0f;
        for (int ky = -kernelRadiusY; ky <= kernelRadiusY; ++ky) {
            for (int kx = -kernelRadiusX; kx <= kernelRadiusX; ++kx) {
                sum += sharedMem[(sharedY + ky) * sharedWidth + (sharedX + kx)];
            }
        }
        output[y * imgWidth + x] = sum;
    }
}

// Convolution kernel using shared memory and vectorized memory loads without textures
__global__ void vectorizedConvolution2D(const float* input, const float* kernel, float* output, 
                                       int imgWidth, int imgHeight, int kernelWidth, int kernelHeight) {
    extern __shared__ float sharedMem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    int kernelRadiusX = kernelWidth / 2;
    int kernelRadiusY = kernelHeight / 2;
    
    int sharedWidth = blockDim.x + 2 * kernelRadiusX;
    
    // Load central region to shared memory
    if (y < imgHeight && x < imgWidth) {
        sharedMem[(ty + kernelRadiusY) * sharedWidth + (tx + kernelRadiusX)] = input[y * imgWidth + x];
    }
    
    // Load halo regions
    if (ty < kernelRadiusY) {
        // Top halo
        int loadY = (blockIdx.y * blockDim.y) - kernelRadiusY + ty;
        if (loadY >= 0 && x < imgWidth) {
            sharedMem[ty * sharedWidth + (tx + kernelRadiusX)] = input[loadY * imgWidth + x];
        } else {
            sharedMem[ty * sharedWidth + (tx + kernelRadiusX)] = 0.0f;
        }
        
        // Bottom halo
        loadY = (blockIdx.y + 1) * blockDim.y + ty;
        if (loadY < imgHeight && x < imgWidth) {
            sharedMem[(blockDim.y + kernelRadiusY + ty) * sharedWidth + (tx + kernelRadiusX)] = input[loadY * imgWidth + x];
        } else {
            sharedMem[(blockDim.y + kernelRadiusY + ty) * sharedWidth + (tx + kernelRadiusX)] = 0.0f;
        }
    }
    
    if (tx < kernelRadiusX) {
        // Left halo
        int loadX = (blockIdx.x * blockDim.x) - kernelRadiusX + tx;
        if (loadX >= 0 && y < imgHeight) {
            sharedMem[(ty + kernelRadiusY) * sharedWidth + tx] = input[y * imgWidth + loadX];
        } else {
            sharedMem[(ty + kernelRadiusY) * sharedWidth + tx] = 0.0f;
        }
        
        // Right halo
        loadX = (blockIdx.x + 1) * blockDim.x + tx;
        if (loadX < imgWidth && y < imgHeight) {
            sharedMem[(ty + kernelRadiusY) * sharedWidth + (blockDim.x + kernelRadiusX + tx)] = input[y * imgWidth + loadX];
        } else {
            sharedMem[(ty + kernelRadiusY) * sharedWidth + (blockDim.x + kernelRadiusX + tx)] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Perform convolution with vectorized loads
    if (x < imgWidth && y < imgHeight) {
        float sum = 0.0f;
        
        for (int ky = 0; ky < kernelHeight; ++ky) {
            int rowOffset = (ty + ky) * sharedWidth + tx;
            int kernelRowOffset = ky * kernelWidth;
            
            int kx = 0;
            
            // Use float4 for processing 4 elements at a time
            for (; kx <= kernelWidth - 4; kx += 4) {
                float4 sharedData;
                sharedData.x = sharedMem[rowOffset + kx];
                sharedData.y = sharedMem[rowOffset + kx + 1];
                sharedData.z = sharedMem[rowOffset + kx + 2];
                sharedData.w = sharedMem[rowOffset + kx + 3];
                
                float4 kernelData;
                kernelData.x = kernel[kernelRowOffset + kx];
                kernelData.y = kernel[kernelRowOffset + kx + 1];
                kernelData.z = kernel[kernelRowOffset + kx + 2];
                kernelData.w = kernel[kernelRowOffset + kx + 3];
                
                sum += sharedData.x * kernelData.x + sharedData.y * kernelData.y + 
                       sharedData.z * kernelData.z + sharedData.w * kernelData.w;
            }
            
            // Use float2 for remaining pairs
            for (; kx <= kernelWidth - 2; kx += 2) {
                float2 sharedData;
                sharedData.x = sharedMem[rowOffset + kx];
                sharedData.y = sharedMem[rowOffset + kx + 1];
                
                float2 kernelData;
                kernelData.x = kernel[kernelRowOffset + kx];
                kernelData.y = kernel[kernelRowOffset + kx + 1];
                
                sum += sharedData.x * kernelData.x + sharedData.y * kernelData.y;
            }
            
            // Process any remaining elements
            for (; kx < kernelWidth; ++kx) {
                sum += sharedMem[rowOffset + kx] * kernel[kernelRowOffset + kx];
            }
        }
        
        output[y * imgWidth + x] = sum;
    }
}

#endif // CNN_CUH