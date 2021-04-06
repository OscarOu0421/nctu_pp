// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#define LIVE 79
#define DEAD 46

void runCudaPart(char **, int);

__global__ void conwayKernel(char* d_grid, char* d_new_grid, int size){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < size && y < size){    
        int count = 0;

        if((x-1)>=0 && (y-1)>=0){
            if(d_grid[(y-1) * size + (x-1)] == LIVE)
                count++;
        }
        if((x-1)>=0){
            if(d_grid[y * size + (x-1)] == LIVE)
                count++;
        }
        if((x-1)>=0 && (y+1)<size){
            if(d_grid[(y+1) * size + (x-1)] == LIVE)
                count++;
        }
        if((y-1)>=0){
            if(d_grid[(y-1) * size + x] == LIVE)
                count++;
        }
        if((y+1)<size){
            if(d_grid[(y+1) * size + x] == LIVE)
                count++;
        }
        if((x+1)<size && (y-1)>=0){
            if(d_grid[(y-1) * size + (x+1)] == LIVE)
                count++;
        }
        if((x+1)<size){
            if(d_grid[y * size + (x+1)] == LIVE)
                count++;
        }
        if((x+1)<size && (y+1)<size){
            if(d_grid[(y+1) * size + (x+1)] == LIVE)
                count++;
        }

        __syncthreads();
        if(d_grid[y * size + x] == LIVE){
            if(count < 2 || count > 3)
                d_new_grid[y * size + x] = DEAD;
            else
                d_new_grid[y * size + x] = LIVE;
        }else{
            if(count == 3)
                d_new_grid[y * size + x] = LIVE;
            else
                d_new_grid[y * size + x] = DEAD;
        }
    }

}

// Main cuda function

void runCudaPart(char *h_grid, char *d_grid, char *d_new_grid, int size) {
    dim3 threadsPerBlock(32, 32);
    int remain = (size % 32 != 0) ? 1 : 0;
    dim3 blocksPerGrid(size / 32 + remain, size / 32 + remain);
    conwayKernel<<<blocksPerGrid, threadsPerBlock>>>(d_grid, d_new_grid, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_grid, d_new_grid, size*size*sizeof(char), cudaMemcpyDeviceToHost);
}
