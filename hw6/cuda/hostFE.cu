#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>


__global__ void convolution(float *inputImage, float *outputImage, float *filter,
                            int imageHeight, int imageWidth, int filterWidth)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int halffilterSize = filterWidth / 2;
    int k, l;
    float sum = 0.0f;

    for (k = -halffilterSize; k <= halffilterSize; k++)
    {
        for (l = -halffilterSize; l <= halffilterSize; l++)
        {
            if(filter[(k + halffilterSize) * filterWidth + l + halffilterSize] != 0)
            {
                if (row + k >= 0 && row + k < imageHeight &&
                    col + l >= 0 && col + l < imageWidth)
                {
                    sum += inputImage[(row + k) * imageWidth + col + l] *
                            filter[(k + halffilterSize) * filterWidth +
                                    l + halffilterSize];
                }
            }
        }
    }
    outputImage[row * imageWidth + col] = sum;
}
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth, float *inputImage, float *outputImage)
{
    float *d_inputImage;
    float *d_outputImage;
    float *d_filter;
    int imageSize = imageHeight * imageWidth;
    int filterSize = filterWidth * filterWidth;

    cudaMalloc((void **)&d_inputImage, imageSize * sizeof(float));
    cudaMalloc((void **)&d_outputImage, imageSize * sizeof(float));
    cudaMalloc((void **)&d_filter,  filterSize * sizeof(float));

    cudaMemcpy(d_inputImage, inputImage, imageSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filterSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlock(imageWidth/threadsPerBlock.x, imageHeight/threadsPerBlock.y);

    convolution<<<numBlock, threadsPerBlock>>>(d_inputImage, d_outputImage, d_filter, imageHeight, imageWidth, filterWidth);
    cudaMemcpy(outputImage, d_outputImage, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_filter);
}