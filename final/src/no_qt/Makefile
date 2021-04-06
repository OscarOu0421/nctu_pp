all:
	g++ omp.cpp -o omp -fopenmp
	g++ thread.cpp -o thread -lpthread
	g++ serial.cpp -o serial
	nvcc cudasrc/main.cpp cudasrc/kernel.cu cudasrc/cuda.cpp -o cuda

.PONY : clean
clean:
	rm -f omp serial thread cuda