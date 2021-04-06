#ifndef CUDA_H
#define CUDA_H

#include<QtWidgets>
#include<QCloseEvent>
#include<iostream>

#define P_SIZE 8 /* pixel size */
#define LIVE 79
#define DEAD 46
#define SHOW_RANGE 50

class gol : public QMainWindow
{
    Q_OBJECT

public:
    gol() {};
    gol(int, QWidget *parent = 0);
    ~gol();
    void nextgen();
    void initpattern();
    void paint_on_window();
    void closeEvent(QCloseEvent*);
    void Free();

private:
    int size;
    char *h_grid;
    char *d_grid, *d_new_grid;
    QLabel *label;
    QPixmap *map;
    QPen *pen;
};

#endif // CUDA_H
