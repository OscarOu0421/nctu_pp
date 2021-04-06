#include "serial.h"

gol::gol(int inputsize, QWidget *parent)
    : QMainWindow(parent), size(inputsize)
{
    char* data = (char*)malloc(size * size * sizeof(char));
    char* tempdata = (char*)malloc(size * size * sizeof(char));
    grid = (char**)malloc(size * sizeof(char*));
    temp = (char**)malloc(size * sizeof(char*));
    for (int i=0; i<size; i++)
    {
        grid[i] = &(data[size * i]);
        temp[i] = &(tempdata[size * i]);
    }
    for(int i=0; i<size; i++)
    {
        for(int j=0; j<size; j++)
        {
            grid[i][j]=DEAD;
            temp[i][j]=DEAD;
        }
    }

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
    for(int i=0; i<size; i++)
    {
        for(int j=0; j<size; j++)
        {
            if(grid[i][j]==LIVE)
            {
                if(neighbours(i,j)<2)
                    setcell(i,j,0);
                else if(neighbours(i,j)>3)
                    setcell(i,j,0);
                else
                    setcell(i,j,1);
            }
            else
            {
                if(neighbours(i,j)==3)
                    setcell(i,j,1);
                else
                    setcell(i,j,0);
            }
        }
    }
    copytogrid();
}

void gol::copytogrid()
{
    for(int i=0; i<size; i++)
        {
            for(int j=0; j<size; j++)
            {
                grid[i][j]=temp[i][j];
            }
        }
}

void gol::setcell(int x, int y, int f)
{
    if(f==1)
        temp[x][y]=LIVE;
    else
        temp[x][y]=DEAD;
}

void gol::initpattern()
{
    //Glider gun
    grid[5][1] = LIVE;
    grid[6][1] = LIVE;
    grid[5][2] = LIVE;
    grid[6][2] = LIVE;

    grid[5][11] = LIVE;
    grid[6][11] = LIVE;
    grid[7][11] = LIVE;
    grid[4][12] = LIVE;
    grid[8][12] = LIVE;
    grid[3][13] = LIVE;
    grid[9][13] = LIVE;
    grid[3][14] = LIVE;
    grid[9][14] = LIVE;
    grid[6][15] = LIVE;
    grid[4][16] = LIVE;
    grid[8][16] = LIVE;
    grid[5][17] = LIVE;
    grid[6][17] = LIVE;
    grid[7][17] = LIVE;
    grid[6][18] = LIVE;

    grid[3][21] = LIVE;
    grid[4][21] = LIVE;
    grid[5][21] = LIVE;
    grid[3][22] = LIVE;
    grid[4][22] = LIVE;
    grid[5][22] = LIVE;
    grid[2][23] = LIVE;
    grid[6][23] = LIVE;
    grid[1][25] = LIVE;
    grid[2][25] = LIVE;
    grid[6][25] = LIVE;
    grid[7][25] = LIVE;

    grid[3][35] = LIVE;
    grid[4][35] = LIVE;
    grid[3][36] = LIVE;
    grid[4][36] = LIVE;

}

int gol::neighbours(int x, int y)
{
    int count=0;
    if((x-1)>=0 && (y-1)>=0 && grid[x-1][y-1]==LIVE)
        count += 1;
    if((x-1)>=0 && grid[x-1][y]==LIVE)
        count += 1;
    if((x-1)>=0 && (y+1)<size && grid[x-1][y+1]==LIVE)
        count += 1;
    if((y-1)>=0 && grid[x][y-1]==LIVE)
        count += 1;
    if((y+1)<size && grid[x][y+1]==LIVE)
        count += 1;
    if((x+1)<size && (y-1)>=0 && grid[x+1][y-1]==LIVE)
        count += 1;
    if((x+1)<size && grid[x+1][y]==LIVE)
        count += 1;
    if((x+1)<size && (y+1)<size && grid[x+1][y+1]==LIVE)
        count += 1;

    return(count);
}

void gol::paint_on_window()
{
    map->fill(QColor(0, 0, 0));
    pen->setColor(QColor(0, 250, 154));
    QPainter painter(map);
    painter.setPen(*pen);
    for(int i=0; i<SHOW_RANGE; i++){
        for(int j=0; j<SHOW_RANGE; j++){
            if(grid[i][j] == LIVE){
                //pen->setColor(QColor(255, 0, 0));
                //painter.setPen(*pen);
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
