// Define this macro so that the texture reference is defined in this file.
// Place this at the very top!
#define DEFINE_TEXTURES
#include "../include/cnn.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>

// Texture reference for 2D image


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
__global__ void optimizedConvolution2D(const float* mask, float* output,
    int imgWidth, int imgHeight, 
    int kernelWidth, int kernelHeight) {
extern __shared__ float sharedMem[];

// Compute block dimensions and kernel radii.
const int blockWidth  = blockDim.x;
const int blockHeight = blockDim.y;
int kernelRadiusX = kernelWidth / 2;
int kernelRadiusY = kernelHeight / 2;

// Dimensions of the shared memory tile (central block + halo)
int sharedWidth  = blockWidth + 2 * kernelRadiusX;
int sharedHeight = blockHeight + 2 * kernelRadiusY;

// Global coordinates of the top-left corner of the tile (including halo)
int x0 = blockIdx.x * blockWidth - kernelRadiusX;
int y0 = blockIdx.y * blockHeight - kernelRadiusY;

// Cooperative loading: each thread loads parts of the tile
for (int j = threadIdx.y; j < sharedHeight; j += blockHeight) {
for (int i = threadIdx.x; i < sharedWidth; i += blockWidth) {
int globalX = x0 + i;
int globalY = y0 + j;
// Use texture fetch; the texture's addressing mode (clamp) handles out-of-bound accesses.
sharedMem[j * sharedWidth + i] = tex2D<float>(texRef, globalX, globalY);
}
}
__syncthreads();

// Compute the output pixel coordinate.
int x = blockIdx.x * blockWidth + threadIdx.x;
int y = blockIdx.y * blockHeight + threadIdx.y;

if (x < imgWidth && y < imgHeight) {
float sum = 0.0f;
// Each thread’s corresponding position in shared memory
int smemX = threadIdx.x + kernelRadiusX;
int smemY = threadIdx.y + kernelRadiusY;
// Loop over the kernel window.
for (int ky = 0; ky < kernelHeight; ky++) {
for (int kx = 0; kx < kernelWidth; kx++) {
float imgVal = sharedMem[(smemY - kernelRadiusY + ky) * sharedWidth + (smemX - kernelRadiusX + kx)];
float maskVal = mask[ky * kernelWidth + kx];
sum += imgVal * maskVal;
}
}
output[y * imgWidth + x] = sum;
}
}




// Convolution kernel using shared memory and vectorized memory loads without textures
__global__ void vectorizedConvolution2D(const float* input, const float* kernel, float* output, 
    int imgWidth, int imgHeight, int kernelWidth, int kernelHeight) {
extern __shared__ float sharedMem[];

const int blockWidth  = blockDim.x;
const int blockHeight = blockDim.y;
int kernelRadiusX = kernelWidth / 2;
int kernelRadiusY = kernelHeight / 2;

int sharedWidth  = blockWidth + 2 * kernelRadiusX;
int sharedHeight = blockHeight + 2 * kernelRadiusY;

int x0 = blockIdx.x * blockWidth - kernelRadiusX;
int y0 = blockIdx.y * blockHeight - kernelRadiusY;

// Cooperative loading with manual clamping.
for (int j = threadIdx.y; j < sharedHeight; j += blockHeight) {
for (int i = threadIdx.x; i < sharedWidth; i += blockWidth) {
int globalX = x0 + i;
int globalY = y0 + j;
// Clamp coordinates to image boundaries.
if (globalX < 0) globalX = 0;
if (globalX >= imgWidth) globalX = imgWidth - 1;
if (globalY < 0) globalY = 0;
if (globalY >= imgHeight) globalY = imgHeight - 1;
sharedMem[j * sharedWidth + i] = input[globalY * imgWidth + globalX];
}
}
__syncthreads();

int x = blockIdx.x * blockWidth + threadIdx.x;
int y = blockIdx.y * blockHeight + threadIdx.y;

if (x < imgWidth && y < imgHeight) {
float sum = 0.0f;
int smemX = threadIdx.x + kernelRadiusX;
int smemY = threadIdx.y + kernelRadiusY;
for (int ky = 0; ky < kernelHeight; ky++) {
for (int kx = 0; kx < kernelWidth; kx++) {
sum += sharedMem[(smemY - kernelRadiusY + ky) * sharedWidth + (smemX - kernelRadiusX + kx)];
}
}
output[y * imgWidth + x] = sum;
}
}
