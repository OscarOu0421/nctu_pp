#ifndef MAINWINDOW_H
#define MAINWINDOW_H

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
    int neighbours(int, int);
    void setcell(int, int, int);
    void initpattern();
    void copytogrid();
    void paint_on_window();
    void closeEvent(QCloseEvent*);

private:
    int size;
    char** grid;
    char** temp;
    QLabel *label;
    QPixmap *map;
    QPen *pen;
};

#endif // MAINWINDOW_H
