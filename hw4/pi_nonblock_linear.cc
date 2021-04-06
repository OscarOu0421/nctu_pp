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
int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    long long int numOfHit = 0;
    int radius = 0x7FFF;
    int radius_square = radius * radius;
    int x, y;

    if (world_rank > 0)
    {
        // TODO: MPI workers
        long long int localSum = 0;
        fast_srand((int) time(NULL), world_rank);
        for (int i = 0; i < tosses/world_size;i++)
        {
            x = fastrand();
            y = fastrand();

            if (x * x + y * y < radius_square)
                localSum += 1;
        }

        int dest = 0;
        int tag = 0;
        MPI_Send(&localSum, 1, MPI_LONG_LONG, dest, tag, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request *requests = new MPI_Request[world_size-1];
        MPI_Status *status = new MPI_Status[world_size-1];
        long long int *recvSum = new long long int[world_size];
        for(int source=1; source<world_size; source++)
        {
            int tag = 0;
            MPI_Irecv(&recvSum[source], 1, MPI_LONG_LONG, source, tag, MPI_COMM_WORLD, &requests[source-1]);
        }

        fast_srand((int) time(NULL), world_rank);
        for (int i = 0; i < tosses/world_size;i++)
        {
            x = fastrand();
            y = fastrand();

            if (x * x + y * y < radius_square)
                numOfHit += 1;
        }

        MPI_Waitall(world_size-1, requests, status);
        for(int source=1; source<world_size; source++)
            numOfHit += recvSum[source];
        
        delete [] requests;
        delete [] status;
    }

    if (world_rank == 0)
    {
        // TODO: PI result
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
