#include<iostream>
#include<stdlib.h>
#include<unistd.h>
#include <pthread.h>
#include <cstring>

#define LIVE 79
#define DEAD 46

using namespace std;

class gol;
struct Parameter
{
    gol* gPointer;
    int id;
};
class gol
{
private:
    int size;
    int threadNum;
    char** grid;
    char** temp;
    pthread_t* thread; 
    pthread_attr_t attr;
    pthread_barrier_t barrier;
    struct Arg
    {
        int start;
        int end;
    };
    Arg* arg;
    Parameter* p;
public:
    gol() {}
    gol(int inputsize, int intputthread)
        :size(inputsize), threadNum(intputthread)
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
                grid[i][j] = DEAD;
                temp[i][j] = DEAD;
            }
        }
    }
    void print();
    static void* nextgen(void*);
    void copytogrid(int, int);
    int neighbours(int, int);
    void setcell(int, int, int);
    void initpattern();
    void setThread();
    void threadWork();
    void threadOver();
    void threadJoin();
    void freeMem();
};
void gol::setThread()
{
    thread = (pthread_t*)malloc(threadNum * sizeof(pthread_t));
    arg = (Arg*)malloc(threadNum * sizeof(Arg));
    p = (Parameter*)malloc(threadNum * sizeof(Parameter));
    int part = size / threadNum;
    for(int i=0; i<threadNum; i++)
    {
        arg[i].start = part * i;
        arg[i].end = part * (i+1);
    }
    pthread_barrier_init (&barrier, NULL, threadNum);
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
}
void gol::threadWork()
{
    for(int i=0; i<threadNum; i++)
    {
        p[i].gPointer = this;
        p[i].id = i;
        if(pthread_create(&thread[i], &attr, gol::nextgen, &p[i])!=0)
        {
            cerr<<"Error thread work, error: "<<errno<<endl;
            exit(1);
        }
    }
}
void gol::threadOver()
{
    pthread_attr_destroy(&attr);
    pthread_barrier_destroy(&barrier);
}
void gol::threadJoin()
{
    void* status;
    for(int i=0; i<threadNum; i++)
        pthread_join(thread[i], &status);
}
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

void* gol::nextgen(void* arg)
{
    Parameter *p = (Parameter*) arg;
    gol* g = p->gPointer;
    int id = p->id;
    int start = g->arg[id].start;
    int end = g->arg[id].end;

    for(int i=start; i<end; i++)
    {
        for(int j=0; j<g->size; j++)
        {
            if(g->grid[i][j]==LIVE)
            {
                if(g->neighbours(i,j)<2)
                    g->setcell(i,j,0);
                else if(g->neighbours(i,j)>3)
                    g->setcell(i,j,0);
                else
                    g->setcell(i,j,1);
            }
            else
            {
                if(g->neighbours(i,j)==3)
                    g->setcell(i,j,1);
                else
                    g->setcell(i,j,0);
            }
        }
    }

    pthread_barrier_wait(&g->barrier);

    g->copytogrid(start, end);

    return NULL;
}

void gol::copytogrid(int start, int end)
{
    for(int i=start; i<end; i++)
    {
        for(int j=0; j<size; j++)
            grid[i][j] = temp[i][j];
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
    free(thread);
    free(arg);
    free(p);
}

int main(int argc, char *argv[])
{
    if(argc < 4){
        printf("Usage: %s [size] [iter] [thread_num]\n", argv[0]);
        return 0;
    }
    int size = atoi(argv[1]);
    int iter = atoi(argv[2]);
    int threadNum = atoi(argv[3]);

    gol g(size, threadNum);
    g.initpattern();
    g.setThread();
    for(int i=0; i<iter; i++)
    {
        // g.print();
        g.threadWork();
        g.threadJoin();
        // cout<<"\r\x1b[40A"<<flush;
        // usleep(75000);
    }
    // g.print();
    // cout<<"\r\x1b[40A"<<flush;
    g.threadOver();
    g.freeMem();

    return(0);
}