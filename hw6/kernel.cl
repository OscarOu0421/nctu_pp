__kernel void convolution(__global float *inputImage, __global float *outputImage, __global float *filter,
                        int imageHeight, int imageWidth, int filterWidth) 
{
    int index = get_global_id(0);
    int row = index /imageWidth;
    int col = index % imageWidth;
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
