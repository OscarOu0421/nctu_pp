#-------------------------------------------------
#
# Project created by QtCreator 2019-03-31T17:08:06
#
#-------------------------------------------------
 
QT       += core gui
 
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
 
TARGET = cuda
TEMPLATE = app
 
# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS
 
# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
 
 
SOURCES += \
        main.cpp \
        cuda.cpp
 
HEADERS += \
        cuda.h
 
CUDA_SOURCES += $$PWD/kernel.cu
CUDA_SDK = /usr/local/cuda
CUDA_DIR = /usr/local/cuda
SYSTEM_NAME = ubuntu
SYSTEM_TYPE = 64
CUDA_ARCH = sm_50
NVCC_OPTIONS = --use_fast_math
 
INCLUDEPATH += $$CUDA_DIR/include
QMAKE_LIBDIR +=$$CUDA_DIR/lib64
CUDA_OBJECTS_DIR = $$PWD
CUDA_LIBS = -lcuda -lcudart
CUDA_INC =$$join(INCLUDEPATH,'" -I"','-I"','"')
#LIBS += $$join(CUDA_LIBS,'.so ', '', '.so')
LIBS += $$CUDA_LIBS
 
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}
QMAKE_CXXFLAGS += -g
