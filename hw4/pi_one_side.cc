#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

unsigned int g_seed;
//Used to seed the generator.
inline void fast_srand(int seed, int world_rank)
{
    g_seed = seed * world_rank;
}
//fastrand routine returns one integer, similar output value range as C lib.
inline int fastrand()
{
    g_seed = (214013*g_seed+2531011);
    return (g_seed>>16)&0x7FFF;
}
int fnz (long long int *recvSum, int size)
{
    int diff = 0;

    for (int i = 0; i < size; i++)
       diff |= (recvSum[i] != 0);

    if (diff)
    {
        int res = 0;
        for (int i = 0; i < size; i++)
        {
            if(recvSum[i] != 0)
                res++;
        }
       return(res == size-1);
    }
    return 0;
}
int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    long long int numOfHit = 0;
    int radius = 0x7FFF;
    int radius_square = radius * radius;
    int x, y;

    if (world_rank == 0)
    {
        // Master
        long long int* recvSum;
        MPI_Alloc_mem(world_size * sizeof(long long int), MPI_INFO_NULL, &recvSum);
        for(int i=0; i<world_size; i++)
            recvSum[i] = 0;

        fast_srand((int) time(NULL), world_rank);
        for (int i = 0; i < tosses/world_size;i++)
        {
            x = fastrand();
            y = fastrand();

            if (x * x + y * y < radius_square)
                numOfHit += 1;
        }

        MPI_Win_create(recvSum, world_size * sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        int ready = 0;
        while (!ready)
        {
            // Without the lock/unlock schedule stays forever filled with 0s
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
            ready = fnz(recvSum, world_size);
            MPI_Win_unlock(0, win);
        }
        for(int i=0; i<world_size; i++)
            numOfHit += recvSum[i];

        // Release the window
        MPI_Win_free(&win);
        // Free the allocated memory
        MPI_Free_mem(recvSum);
    }
    else
    {
        // Workers
        long long int localSum = 0;
        fast_srand((int) time(NULL), world_rank);
        for (int i = 0; i < tosses/world_size;i++)
        {
            x = fastrand();
            y = fastrand();

            if (x * x + y * y < radius_square)
                localSum += 1;
        }
        
        // Worker processes do not expose memory in the window
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        // Register with the master
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Put(&localSum, 1, MPI_LONG_LONG, 0, world_rank, 1, MPI_LONG_LONG, win);
        MPI_Win_unlock(0, win);

        // Release the window
        MPI_Win_free(&win);
    }

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = 4.0 * numOfHit / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}