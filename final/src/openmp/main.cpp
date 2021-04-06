#include <QApplication>
#include <QtWidgets>
#include <unistd.h>
#include "openmp.h"

int main(int argc, char *argv[])
{
    if(argc < 3){
        printf("Usage: %s [size] [iter]\n", argv[0]);
        return 0;
    }
    if(atoi(argv[1]) < SHOW_RANGE){
        printf("size must over than %d!\n", SHOW_RANGE);
        return 0;
    }
    int size = atoi(argv[1]);
    int iter = atoi(argv[2]);

    QApplication app(argc, argv);
    gol g(size, 0);
    g.initpattern();
    g.paint_on_window();
    g.show();
    //sleep(6); /* need time to open the recorder*/
    for(int i=0; i<iter; i++){
        g.nextgen();
        g.paint_on_window();
        g.update();
        app.processEvents();
    }
    /* stop, exit when I push ctrl+C */
    //while(1){};
    return app.exec();
}
