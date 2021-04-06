#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

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
    MPI_Status status;
    int radius = 0x7FFF;
    int radius_square = radius * radius;
    int x, y;

    // TODO: binary tree redunction
    long long int localSum = 0;
    fast_srand((int) time(NULL), world_rank);
    for (int i = 0; i < tosses/world_size;i++)
    {
        x = fastrand();
        y = fastrand();

        if (x * x + y * y < radius_square)
            localSum += 1;
    }

    int tag = 0;
    int loop = 0;
    int index = world_rank / pow(2, loop);
    bool even = (index % 2 == 0) ? true : false;
    while(even)
    {
        int source = (index + 1) * pow(2, loop);
        long long int recvSum = 0;
        MPI_Recv(&recvSum, 1, MPI_LONG_LONG, source, tag, MPI_COMM_WORLD, &status);
        // printf("rank %d recv %d\n",world_rank, source);
        localSum += recvSum;

        loop++;
        index = world_rank / pow(2, loop);
        even = (index % 2 == 0) ? true : false;
        if(loop == log2(world_size))
            break;
    }
    if(world_rank > 0 )
    {
        int dest = (index - 1) * pow(2, loop);
        MPI_Send(&localSum, 1, MPI_LONG_LONG, dest, tag, MPI_COMM_WORLD);
        // printf("rank %d send %d\n", world_rank, dest);
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4.0 * localSum / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
