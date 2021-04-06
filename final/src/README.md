Conway's Game of Life
===

## File Structure
```
conway/
├── serial/
│   ├── Makefile
│   ├── main.cpp
│   ├── serial.cpp
│   ├── serial.h
│   └── serial.pro
├── openmp/
│   ├── Makefile
│   ├── main.cpp
│   ├── openmp.cpp
│   ├── openmp.h
│   └── openmp.pro
├── cuda/
│   ├── Makefile
│   ├── main.cpp
│   ├── cuda.cpp
│   ├── cuda.h
│   ├── kernel.cu
│   ├── kernel.h
│   └── cuda.pro
└── no_qt/
    ├── Makefile
    ├── thread.cpp
    ├── omp.cpp
    ├── serial.cpp
    └── cudasrc/
        ├── main.cpp
        ├── cuda.cpp
        ├── cuda.h
        ├── kernel.cu
        └── kernel.h

```

Build and Run
---
There are four folders in our project.
In `serial/`, `openmp/` and `cuda/` are the implementations with Qt library, which can show the part of the cell's stat (50\*50) with graphic.
In `no_qt/` is the implementation without Qt library, gather with versions of serial, pthread, OpenMP and CUDA.

> note that `serial/`, `openmp/` and `cuda/` can only run on port `37094` where is the only port with Qt library.

for build and run in `serial/`:
```
$ cd serial; make; ./serial [size] [iter]
```
for build and run in `openmp/`:
```
$ cd openmp; make; ./openmp [size] [iter]
```
for build and run in `cuda/`:
```
$ cd cuda; make; ./cuda [size] [iter]
```
for build and run and measure execution time in `no_qt/`:
```
$ cd no_qt; make
$ time ./serial [size] [iter]
$ time ./thread [size] [iter] [thread_num]
$ time ./omp [size] [iter]
$ time ./cuda [size] [iter]
```
