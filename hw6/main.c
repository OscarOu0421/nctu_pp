#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include "CycleTimer.h"
#include "helper.h"
#include "hostFE.h"
#include "bmpfuncs.h"
#include "serialConv.h"

void usage(const char *progname)
{
   printf("Usage: %s [options]\n", progname);
   printf("Program Options:\n");
   printf("  -f  --filter  <INT>    Use which filter (0, 1, 2)\n");
   printf("  -?  --help             This message\n");
}

int compare(const void *a, const void *b)
{
   double *x = (double *)a;
   double *y = (double *)b;
   if (*x < *y)
      return -1;
   else if (*x > *y)
      return 1;
   return 0;
}

int main(int argc, char **argv)
{
   int i, j;

   // Rows and columns in the input image
   int imageHeight;
   int imageWidth;

   double start_time, end_time;

   const char *inputFile = "input.bmp";
   const char *outputFile = "output.bmp";
   const char *refFile = "ref.bmp";
   char *filterFile = "filter1.csv";

   // parse commandline options ////////////////////////////////////////////
   int opt;
   static struct option long_options[] = {
       {"filter", 1, 0, 'f'},
       {"help", 0, 0, '?'},
       {0, 0, 0, 0}};

   while ((opt = getopt_long(argc, argv, "f:?", long_options, NULL)) != EOF)
   {

      switch (opt)
      {
      case 'f':
      {
         int idx = atoi(optarg);
         if (idx == 2)
            filterFile = "filter2.csv";
         else if (idx == 3)
            filterFile = "filter3.csv";

         break;
      }
      case '?':
      default:
         usage(argv[0]);
         return 1;
      }
   }
   // end parsing of commandline options

   // read filter data
   int filterWidth;
   float *filter = readFilter(filterFile, &filterWidth);

   // Homegrown function to read a BMP from file
   float *inputImage = readImage(inputFile, &imageWidth, &imageHeight);
   // Size of the input and output images on the host
   int dataSize = imageHeight * imageWidth * sizeof(float);
   // Output image on the host
   float *outputImage = (float *)malloc(dataSize);

   // Output image of reference on the host
   float *refImage = NULL;
   refImage = (float *)malloc(dataSize);
   memset(refImage, 0, dataSize);

   serialConv(filterWidth, filter, imageHeight, imageWidth, inputImage, refImage);

   // helper init CL
   cl_program program;
   cl_device_id device;
   cl_context context;
   initCL(&device, &context, &program);

   hostFE(filterWidth, filter, imageHeight, imageWidth, inputImage, outputImage,
             &device, &context, &program);

   // Write the output image to file

   int diff_counter = 0;
   for (i = 0; i < imageHeight; i++)
   {
      for (j = 0; j < imageWidth; j++)
      {
         if (abs(outputImage[i * imageWidth + j] - refImage[i * imageWidth + j]) > 10)
         {
            diff_counter += 1;
         }
      }
   }

   float diff_ratio = (float)diff_counter / (imageHeight * imageWidth);
   printf("Diff ratio: %f\n", diff_ratio);

   if (diff_ratio > 0.1)
   {
      printf("\n\033[31mFAILED:\tResults are incorrect!\033[0m\n");
      return -1;
   }
   else
   {
      printf("\n\033[32mPASS\033[0m\n");
   }

   return 0;
}
