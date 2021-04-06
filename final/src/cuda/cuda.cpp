#include "cuda.h"
#include "kernel.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
gol::gol(int inputsize, QWidget *parent)
    : QMainWindow(parent), size(inputsize)
{
    h_grid = (char *)malloc(size * size * sizeof(char));
    cudaMalloc((void **)&d_grid, size * size * sizeof(char));
    cudaMalloc((void **)&d_new_grid, size * size * sizeof(char));
    initpattern();
    cudaMemcpy(d_grid, h_grid, size * size * sizeof(char), cudaMemcpyHostToDevice);

    label = new QLabel();
    map = new QPixmap(SHOW_RANGE*P_SIZE, SHOW_RANGE*P_SIZE);
    pen = new QPen();
    pen->setWidthF(P_SIZE);
}

gol::~gol()
{

}

void gol::nextgen()
{
    runCudaPart(h_grid, d_grid, d_new_grid, size);
    
    char *tmp;
    tmp = d_new_grid;
    d_new_grid = d_grid;
    d_grid = tmp;
}

void gol::initpattern()
{
    //Glider gun
    h_grid[5 * size + 1] = LIVE;
    h_grid[6 * size + 1] = LIVE;
    h_grid[5 * size + 2] = LIVE;
    h_grid[6 * size + 2] = LIVE;

    h_grid[5 * size + 11] = LIVE;
    h_grid[6 * size + 11] = LIVE;
    h_grid[7 * size + 11] = LIVE;
    h_grid[4 * size + 12] = LIVE;
    h_grid[8 * size + 12] = LIVE;
    h_grid[3 * size + 13] = LIVE;
    h_grid[9 * size + 13] = LIVE;
    h_grid[3 * size + 14] = LIVE;
    h_grid[9 * size + 14] = LIVE;
    h_grid[6 * size + 15] = LIVE;
    h_grid[4 * size + 16] = LIVE;
    h_grid[8 * size + 16] = LIVE;
    h_grid[5 * size + 17] = LIVE;
    h_grid[6 * size + 17] = LIVE;
    h_grid[7 * size + 17] = LIVE;
    h_grid[6 * size + 18] = LIVE;

    h_grid[3 * size + 21] = LIVE;
    h_grid[4 * size + 21] = LIVE;
    h_grid[5 * size + 21] = LIVE;
    h_grid[3 * size + 22] = LIVE;
    h_grid[4 * size + 22] = LIVE;
    h_grid[5 * size + 22] = LIVE;
    h_grid[2 * size + 23] = LIVE;
    h_grid[6 * size + 23] = LIVE;
    h_grid[1 * size + 25] = LIVE;
    h_grid[2 * size + 25] = LIVE;
    h_grid[6 * size + 25] = LIVE;
    h_grid[7 * size + 25] = LIVE;

    h_grid[3 * size + 35] = LIVE;
    h_grid[4 * size + 35] = LIVE;
    h_grid[3 * size + 36] = LIVE;
    h_grid[4 * size + 36] = LIVE;

}

void gol::paint_on_window()
{
    map->fill(QColor(0, 0, 0));
    pen->setColor(QColor(0, 250, 154));
    QPainter painter(map);
    painter.setPen(*pen);
    for(int i=0; i<SHOW_RANGE; i++){
        for(int j=0; j<SHOW_RANGE; j++){
            if(h_grid[i * size + j] == LIVE){
                painter.drawPoint(j*P_SIZE, i*P_SIZE);
            }
        }
    }
    painter.end();
    label->setPixmap(*map);
    this->setCentralWidget(label);
}

void gol::closeEvent(QCloseEvent *event)
{
    event->accept();
    exit(0);
}

void gol::Free()
{
    free(h_grid);
    cudaFree(d_grid);
    cudaFree(d_new_grid);
}
