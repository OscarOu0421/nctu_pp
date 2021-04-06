#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int imageSize = imageHeight * imageWidth;
    char *sourceStr = readSource("kernel.cl");

    cl_command_queue commandQueue = clCreateCommandQueue(*context, *device, 0, NULL);

    cl_mem inputImageMem = clCreateBuffer(*context, CL_MEM_READ_ONLY, imageSize * sizeof(float), NULL, NULL);
    cl_mem filterMem = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize * sizeof(float), NULL, NULL);
    cl_mem outputImageMem = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize * sizeof(float), NULL, NULL);

    clEnqueueWriteBuffer(commandQueue, inputImageMem, CL_TRUE, 0,imageSize * sizeof(float), inputImage, 0, NULL, NULL);
    clEnqueueWriteBuffer(commandQueue, filterMem, CL_TRUE, 0,filterSize * sizeof(float), filter, 0, NULL, NULL);

    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputImageMem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&outputImageMem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&filterMem);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&imageHeight);
    clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&filterWidth);

    size_t globalItemSize = imageSize;
    size_t localItemSize = 64;
    clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);
    clEnqueueReadBuffer(commandQueue, outputImageMem, CL_TRUE, 0, imageSize * sizeof(float), outputImage, 0, NULL, NULL);
}