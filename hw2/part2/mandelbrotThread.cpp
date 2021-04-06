#include <stdio.h>
#include <thread>

#include "CycleTimer.h"

using namespace std;

typedef struct
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
} WorkerArgs;

extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);

//
// workerThreadStart --
//
// Thread entrypoint.
void workerThreadStart(WorkerArgs *const args)
{

    // TODO FOR PP STUDENTS: Implement the body of the worker
    // thread here. Each thread should make a call to mandelbrotSerial()
    // to compute a part of the output image.  For example, in a
    // program that uses two threads, thread 0 could compute the top
    // half of the image and thread 1 could compute the bottom half.
    
    // double startTime = CycleTimer::currentSeconds();
    
    int startRow, numRows;

    printf("Hello world from thread %d\n", args->threadId);

    if(args->numThreads == 2)
    {
        int region = args->height / args->numThreads;
        startRow = args->threadId * region;
        numRows = region;
    }
    else if(args->x0 == -2)
    {
        /*  view 1 */
        int relax = args->numThreads - 3;
        if(relax > 10)
            relax = 10;
        else if(relax < 0)
            relax = 0;
        int first = 500 - relax * 50;
        int middle = 200 + relax * 100;
        int last = 500 - relax * 50;

        if(args->threadId == 0)
        {
            /*  first thread */
            startRow = 0;
            numRows = first;
        }
        else if(args->threadId == args->numThreads -1)
        {
            /*  last thread  */
            int remainRow = middle % (args->numThreads -2);
            startRow = first + middle - remainRow;
            numRows = last + remainRow;
        }
        else
        {
            /*  middle region */
            int region = middle / (args->numThreads - 2);
            startRow = first + region * (args->threadId - 1);
            numRows = region;
        }
    }
    else
    {
        /*  view2   */
        int relax = args->numThreads - 3;
        if(relax > 9)
            relax = 9;
        else if(relax < 0)
            relax = 0;
        int first = 300 + relax * 100;
        int middle = 450 - relax * 50;
        int last = 450 - relax * 50;

        if(args->threadId == args->numThreads -2)
        {
            /*  last second thread */
            int remainRow = first % (args->numThreads -2);
            startRow = first - remainRow;
            numRows = middle + remainRow;
        }
        else if(args->threadId == args->numThreads -1)
        {
            /*  last thread  */
            startRow = first + middle;
            numRows = last;
        }
        else
        {
            /*  first region */
            int region = first / (args->numThreads - 2);
            startRow = 0 + region * args->threadId;
            numRows = region;
        }
    }

    mandelbrotSerial(args->x0, args->y0, args->x1, args->y1, args->width, args->height, startRow, numRows, args->maxIterations, args->output);

    // double endTime = CycleTimer::currentSeconds();
    // printf("Thread %d :  %.3f ms start row: %d numrber row: %d\n", args->threadId, (endTime - startTime)*1000, startRow, numRows);
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];


    for (int i = 0; i < numThreads; i++)
    {
        // TODO FOR PP STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
 
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;

        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < numThreads; i++)
    {

        workers[i] = thread(workerThreadStart, &args[i]);
    }

    workerThreadStart(&args[0]);

    // join worker threads
    for (int i = 1; i < numThreads; i++)
    {
        workers[i].join();
    }
}
