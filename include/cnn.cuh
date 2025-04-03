#ifndef CNN_CUH
#define CNN_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>

// Texture reference declaration
extern texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// Function prototypes (declarations only)
void cpuConvolution2D(const float* input, const float* kernel, float* output, 
                      int imgWidth, int imgHeight, int kernelWidth, int kernelHeight);
__global__ void naiveConvolution2D(const float* input, const float* kernel, float* output,
                                     int imgWidth, int imgHeight, int kernelWidth, int kernelHeight);
__global__ void optimizedConvolution2D(float* output, int imgWidth, int imgHeight, 
                                         int kernelWidth, int kernelHeight);
__global__ void vectorizedConvolution2D(const float* input, const float* kernel, float* output, 
                                          int imgWidth, int imgHeight, int kernelWidth, int kernelHeight);

#endif // CNN_CUH
