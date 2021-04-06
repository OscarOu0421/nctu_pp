#include <mpi.h>
#include <cstdio>
#include <iostream>

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

using namespace std;

int taskId, numTasks, NRA, NCA, NCB;
int** a;
int** b;
int** c;
// *********************************************
// ** ATTENTION: YOU CANNOT MODIFY THIS FILE. **
// *********************************************


int **Allocate2dInt(int rows, int cols) {
    int *data = (int *)malloc(rows*cols*sizeof(int));
    int **array= (int **)malloc(rows*sizeof(int*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);

    return array;
}
// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from stdin
//
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &taskId);
    MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
    int numWorkers = numTasks-1;
    if(taskId == MASTER)
    {
        /*  Construct matrix    */
        cin>>*n_ptr>>*m_ptr>>*l_ptr;
        NRA = *n_ptr;
        NCA = *m_ptr;
        NCB = *l_ptr;
        for(int dest=1; dest<=numWorkers; dest++)
        {
            MPI_Send(&NRA, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&NCA, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&NCB, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
        }
        a_mat_ptr = Allocate2dInt(NRA, NCA);
        a = a_mat_ptr;
        b_mat_ptr = Allocate2dInt(NCA, NCB);
        b = b_mat_ptr;
        for(int row=0; row<NRA; row++)
        {
            for(int col=0; col<NCA; col++)
                cin>>a[row][col];
        }
        for(int row=0; row<NCA; row++)
        {
            for(int col=0; col<NCB; col++)
                cin>>b[row][col];
        }
    }
    else
    {
        /*  Worker recv data    */
        MPI_Status status;
        MPI_Recv(&NRA, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&NCA, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&NCB, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        a_mat_ptr = Allocate2dInt(NRA, NCA);
        a = a_mat_ptr;
        b_mat_ptr = Allocate2dInt(NCA, NCB);
        b = b_mat_ptr;
    }
}

// Just matrix multiplication (your should output the result in this function)
// 
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat)
{
    if(taskId == MASTER)
    {
        /*  Send data to worker */
        MPI_Status status;
        int numWorkers = numTasks;
        int avgRow = NRA / numWorkers;
        int extraRow = NRA % numWorkers;
        int rows = (0 < extraRow) ? avgRow+1 : avgRow;
        int offset = 0 + rows;
        
        for (int dest=1; dest<numWorkers; dest++)
        {
            rows = (dest < extraRow) ? avgRow+1 : avgRow;  	
            MPI_Send(&offset, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&(a[offset][0]), rows*NCA, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&(b[0][0]), NCA*NCB, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            offset = offset + rows;
        }

        rows = (0 < extraRow) ? avgRow+1 : avgRow;
        c = Allocate2dInt(NRA, NCB);
        for (int k=0; k<NCB; k++)
        {
            for (int i=0; i<rows; i++)
            {
                c[i][k] = 0;
                for (int j=0; j<NCA; j++)
                    c[i][k] = c[i][k] + a[i][j] * b[j][k];
            }
        }

        // /* Receive results from worker tasks */
        for (int source=1; source<numWorkers; source++)
        {
            MPI_Recv(&offset, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows*NCB, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);
        }

        /* Print results */
        for (int i=0; i<NRA; i++)
        { 
            for (int j=0; j<NCB; j++)
            {
                if(j == NCB-1)
                    printf("%d", c[i][j]);
                else
                    printf("%d ", c[i][j]);
            } 
            printf("\n");
        }
    }
    else
    {
        MPI_Status status;
        int offset, rows;
        MPI_Recv(&offset, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&(a[offset][0]), rows*NCA, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&(b[0][0]), NCA*NCB, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);

        c = Allocate2dInt(rows, NCB);
        for (int k=0; k<NCB; k++)
        {
            for (int i=0; i<rows; i++)
            {
                c[i][k] = 0;
                for (int j=0; j<NCA; j++)
                    c[i][k] = c[i][k] + a[i+offset][j] * b[j][k];
            }
        }
        
        MPI_Send(&offset, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&c[0][0], rows*NCB, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);

    }
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat)
{
    free(a[0]);
    free(a);
    free(b[0]);
    free(b);
    free(c[0]);
    free(c);
}

int main () {
    int n, m, l;
    int *a_mat, *b_mat;

    MPI_Init(NULL, NULL);
    double start_time = MPI_Wtime();

    construct_matrices(&n, &m, &l, &a_mat, &b_mat);
    matrix_multiply(n, m, l, a_mat, b_mat);
    destruct_matrices(a_mat, b_mat);

    double end_time = MPI_Wtime();
    MPI_Finalize();
    printf("MPI running time: %lf Seconds\n", end_time - start_time);

    return 0;
}