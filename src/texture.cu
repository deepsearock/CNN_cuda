#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>

// Macro for error checking
#define CUDA_CHECK(call) {                                              \
    cudaError_t err = call;                                             \
    if(err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                cudaGetErrorString(err));                               \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

// Declare a texture reference for the input image
texture<int, cudaTextureType2D, cudaReadModeElementType> texRef;

// Convolution kernel using texture and shared memory
__global__ void convolution2DKernel(int *output, int imageWidth, int imageHeight, int maskWidth, int maskRadius) {
    // Dynamically allocated shared memory for the image tile including halos.
    extern __shared__ int sharedMem[];
    
    // Thread indices in the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global pixel coordinates
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    
    // Calculate dimensions of shared memory tile (blockDim + halo)
    int sharedWidth = blockDim.x + 2 * maskRadius;
    int sharedHeight = blockDim.y + 2 * maskRadius;
    
    // Coordinates in shared memory for the central part (offset by maskRadius)
    int shared_x = tx + maskRadius;
    int shared_y = ty + maskRadius;
    
    // 1. Load the central data into shared memory from texture memory.
    if (row < imageHeight && col < imageWidth) {
        sharedMem[shared_y * sharedWidth + shared_x] = tex2D(texRef, col, row);
    } else {
        sharedMem[shared_y * sharedWidth + shared_x] = 0;
    }
    
    // 2. Load halo regions.
    // Left halo
    if (tx < maskRadius) {
        int halo_col = col - maskRadius;
        int value = (halo_col >= 0 && row < imageHeight) ? tex2D(texRef, halo_col, row) : 0;
        sharedMem[shared_y * sharedWidth + tx] = value;
    }
    // Right halo
    if (tx >= blockDim.x - maskRadius) {
        int halo_col = col + maskRadius;
        int value = (halo_col < imageWidth && row < imageHeight) ? tex2D(texRef, halo_col, row) : 0;
        sharedMem[shared_y * sharedWidth + shared_x + maskRadius] = value;
    }
    // Top halo
    if (ty < maskRadius) {
        int halo_row = row - maskRadius;
        int value = (halo_row >= 0 && col < imageWidth) ? tex2D(texRef, col, halo_row) : 0;
        sharedMem[ty * sharedWidth + shared_x] = value;
    }
    // Bottom halo
    if (ty >= blockDim.y - maskRadius) {
        int halo_row = row + maskRadius;
        int value = (halo_row < imageHeight && col < imageWidth) ? tex2D(texRef, col, halo_row) : 0;
        sharedMem[(shared_y + maskRadius) * sharedWidth + shared_x] = value;
    }
    
    // Corner halos: top-left, top-right, bottom-left, bottom-right
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
    
    // 3. Apply the convolution operation using the shared memory tile.
    if (row < imageHeight && col < imageWidth) {
        int output_value = 0;
        // Iterate over the convolution mask.
        for (int i = -maskRadius; i <= maskRadius; i++) {
            for (int j = -maskRadius; j <= maskRadius; j++) {
                int image_value = sharedMem[(shared_y + i) * sharedWidth + (shared_x + j)];
                output_value += image_value;
            }
        }
        // For an averaging filter, divide by the area of the mask.
        output_value /= ((2 * maskRadius + 1) * (2 * maskRadius + 1));
        output[row * imageWidth + col] = output_value;
    }
}

// CPU implementation of the 2D convolution
void convolution2D_CPU(const int *h_image, int *h_output, int imageWidth, int imageHeight, int maskWidth) {
    int maskRadius = maskWidth / 2;
    // Loop over every pixel in the image
    for (int row = 0; row < imageHeight; row++) {
        for (int col = 0; col < imageWidth; col++) {
            int sum = 0;
            // Loop over the mask
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
            // Average the sum over the mask area
            h_output[row * imageWidth + col] = sum / ((2 * maskRadius + 1) * (2 * maskRadius + 1));
        }
    }
}

int main(int argc, char **argv) {
    // Set default dimensions: 512x512 image with a 3x3 mask.
    int dimX = 512, dimY = 512, dimK = 3;
    
    // Parse command-line arguments: -i <dimX> -j <dimY> -k <dimK>
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0)
            dimX = atoi(argv[++i]);
        else if (strcmp(argv[i], "-j") == 0)
            dimY = atoi(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0)
            dimK = atoi(argv[++i]);
    }
    
    // Compute total number of pixels in the image.
    int imageSize = dimX * dimY;
    
    // Allocate host memory for the image and the outputs.
    int *h_image = (int *)malloc(sizeof(int) * imageSize);
    int *h_output_gpu = (int *)malloc(sizeof(int) * imageSize);
    int *h_output_cpu = (int *)malloc(sizeof(int) * imageSize);
    
    // Randomly generate pixel values between 0 and 15.
    for (int i = 0; i < imageSize; i++) {
        h_image[i] = rand() % 16;
    }
    
    // Define mask parameters.
    int maskWidth = dimK;
    int maskRadius = maskWidth / 2;
    
    // 1. Allocate device memory for the image as a CUDA array (to be used for texture binding).
    cudaArray *d_imageArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
    CUDA_CHECK(cudaMallocArray(&d_imageArray, &channelDesc, dimX, dimY));
    
    // 2. Copy host image to the CUDA array (device memory).
    CUDA_CHECK(cudaMemcpy2DToArray(d_imageArray, 0, 0, h_image, dimX * sizeof(int),
                                   dimX * sizeof(int), dimY, cudaMemcpyHostToDevice));
    
    // 3. Bind the CUDA array to the texture reference.
    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.filterMode = cudaFilterModePoint;
    texRef.normalized = false;
    CUDA_CHECK(cudaBindTextureToArray(texRef, d_imageArray, channelDesc));
    
    // 4. Allocate device memory for the output.
    int *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int) * imageSize));
    
    // 5. Set up thread block and grid dimensions.
    dim3 blockDim(16, 16);
    dim3 gridDim((dimX + blockDim.x - 1) / blockDim.x, (dimY + blockDim.y - 1) / blockDim.y);
    
    // Calculate the size of shared memory needed for each block.
    int sharedMemSize = (blockDim.x + 2 * maskRadius) * (blockDim.y + 2 * maskRadius) * sizeof(int);
    
    // 6. Invoke the convolution kernel.
    convolution2DKernel<<<gridDim, blockDim, sharedMemSize>>>(d_output, dimX, dimY, maskWidth, maskRadius);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check for any kernel errors.
    CUDA_CHECK(cudaGetLastError());
    
    // 7. Copy the results from device to host.
    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, sizeof(int) * imageSize, cudaMemcpyDeviceToHost));
    
    // 8. Clean up: Unbind texture and free device memory.
    CUDA_CHECK(cudaUnbindTexture(texRef));
    CUDA_CHECK(cudaFreeArray(d_imageArray));
    CUDA_CHECK(cudaFree(d_output));
    
    // Compute CPU convolution for comparison.
    convolution2D_CPU(h_image, h_output_cpu, dimX, dimY, maskWidth);
    
    // Compare GPU and CPU results.
    int errorCount = 0;
    for (int i = 0; i < imageSize; i++) {
        if (abs(h_output_gpu[i] - h_output_cpu[i]) > 1e-5) {
            errorCount++;
            if (errorCount < 10) {
                printf("Mismatch at index %d: GPU = %d, CPU = %d\n", i, h_output_gpu[i], h_output_cpu[i]);
            }
        }
    }
    if (errorCount == 0) {
        printf("GPU and CPU results match.\n");
    } else {
        printf("Total mismatches: %d\n", errorCount);
    }
    
    // (Optional) Print a sample of the GPU output for verification.
    printf("Sample convolution results (GPU):\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_output_gpu[i]);
    }
    printf("\n");
    
    // Free host memory.
    free(h_image);
    free(h_output_gpu);
    free(h_output_cpu);
    
    return 0;
}
