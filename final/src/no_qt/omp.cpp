#include<iostream>
#include<stdlib.h>
#include<unistd.h>
#include <omp.h>

#define LIVE 79
#define DEAD 46

using namespace std;

class gol
{
private:
    int size;
    char** grid;
    char** temp;
public:
    gol() {}
    gol(int inputsize)
        :size(inputsize)
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
    }
    void print();
    void nextgen();
    int neighbours(int, int);
    void setcell(int, int, int);
    void initpattern();
    void copytogrid();
    void freeMem();
};
void gol::print()
{
    for(int i=0; i<size; i++)
    {
        for(int j=0;j<size;j++)
        {
            cout<<grid[i][j]<<" ";
        }
        cout<<"\n";
    }
}

void gol::nextgen()
{
    #pragma omp parallel for
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
    #pragma omp parallel for
    for(int i=0; i<size; i++)
        {
            for(int j=0; j<size; j++)
            {
                grid[i][j]=temp[i][j];
            }
        }
}

int gol::neighbours(int x, int y)
{
    int count=0;
    if((x-1)>=0 && (y-1)>=0 && grid[x-1][y-1]==LIVE)
        count += 1;
    if((x-1)>=0 && grid[x-1][y]==LIVE)
        count += 1;
    if((x-1)>=0 && grid[x-1][y+1]==LIVE)
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
void gol::freeMem()
{
    free(grid[0]);
    free(temp[0]);
    free(grid);
    free(temp);
}

int main(int argc, char *argv[])
{
    if(argc < 3){
        printf("Usage: %s [size] [iter]\n", argv[0]);
        return 0;
    }
    int size = atoi(argv[1]);
    int iter = atoi(argv[2]);
    gol g(size);
    g.initpattern();

    for(int i=0; i<iter; i++)
    {
        // g.print();
        g.nextgen();
        // cout<<"\r\x1b[40A"<<flush;
        // usleep(75000);
    }
    // g.print();
    // cout<<"\r\x1b[40A"<<flush;
    g.freeMem();

    return(0);
}