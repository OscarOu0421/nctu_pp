#include <iostream>
#include <pthread.h>
#include <string>
#include <stdlib.h> //rand_r
#include <time.h>

using namespace std;

/*  argument for thread do function */
struct Arg
{
    int id;
    int start;
    int end;
    long long int* numOfHit;
};

/*  mutex for sum   */
pthread_mutex_t mutexSum;

void *CalPi(void* arg)
{
    Arg* data = (Arg*)arg;
    int id = data->id;
    int start = data->start;
    int end = data->end;

    long long int* numOfHit = data->numOfHit;
    long long int localSum = 0;

    unsigned int seed = 2021;
    for(int i=start; i<end; i++)
    {
        double x = (double)rand_r(&seed)/RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed)/RAND_MAX * 2.0 - 1.0;
        double dis = x*x + y*y;

        if(dis <= 1.0)
            localSum++;
    }

    /*  access share variable   */
    pthread_mutex_lock(&mutexSum);
    *numOfHit += localSum;
    pthread_mutex_unlock(&mutexSum);

    pthread_exit((void*)0);
}
int main(int argc, char* const argv[])
{
    if(argc != 3)
    {
        cerr<<"there must are two argument, such as ./pi.out 4 10000"<<endl;
        return 1;
    }

    int threadNum = atoi(argv[1]);
    long long int numOfTosses = atoll(argv[2]);
    int part = numOfTosses / threadNum;
    Arg arg[threadNum];

    /*  share variable number of hit   */
    long long int *numOfHit = new long long int;
    *numOfHit = 0;

    pthread_t thread[threadNum];

    pthread_mutex_init(&mutexSum, NULL);

    /*  set thread attribute joinable   */
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);


    for(int i=0; i<threadNum; i++)
    {
        /*  set argument with each thread   */
        arg[i].id = i;
        arg[i].start = part * i;
        arg[i].end = part * (i+1);
        arg[i].numOfHit = numOfHit;

        if(pthread_create(&thread[i], &attr, CalPi, (void*) &arg[i])!=0)
        {
            cerr<<"create thread error"<<endl;
            return 1;
        }
    }
    
    pthread_attr_destroy(&attr);

    void* status;
    for(int i = 0; i<threadNum; i++)
        pthread_join(thread[i], &status);

    pthread_mutex_destroy(&mutexSum);

    double pi = 4.0 * (*numOfHit) / (double)numOfTosses;
    cout.precision(7);
    cout<<fixed<<pi<<endl;

    delete numOfHit;
    
    return 0;
}