default: conv

CC = gcc-10
FLAGS = -O3 -lOpenCL -m64 -ffloat-store -w -g

OBJS = main.o bmpfuncs.o hostFE.o serialConv.o helper.o

conv: $(OBJS)
	$(CC) -o $@ $(OBJS) $(FLAGS)

%.o: %.c
	$(CC) -c $(FLAGS) $< -o $@

clean:
	rm -f conv *.o output.bmp ref.bmp

# NVCC = nvcc
# CXX=g++
# #CXX = gcc-10
# #CXXFLAGS=-Iobjs/ -O3 -std=c++17 -Wall -g -fPIC -lm
# CXXFLAGS=-Iobjs/ -O3 -m64 -ffloat-store -w -g

# APP_NAME=conv
# OBJDIR=objs

# CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' 
# CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3

# #PPM_OBJ=$(addprefix $(OBJDIR)/, $(subst $(COMMONDIR)/,, $(PPM_CXX:.cpp=.o)))

# default: $(APP_NAME)

# .PHONY: dirs clean

# dirs:
# 		/bin/mkdir -p $(OBJDIR)/

# clean:
# 		/bin/rm -rf $(OBJDIR) *~ $(APP_NAME)

# OBJS=$(OBJDIR)/main.o $(OBJDIR)/bmpfuncs.o $(OBJDIR)/hostFE.o $(OBJDIR)/serialConv.o $(OBJDIR)/helper.o

# $(APP_NAME): dirs $(OBJS)
# 		$(NVCC) ${CUDA_LINK_FLAGS} -o $@ $(OBJS)

# $(OBJDIR)/%.o: %.c
# 		$(CXX) $< $(CXXFLAGS) -c -o $@

# #$(OBJDIR)/%.o: $(COMMONDIR)/%.cpp
# #	$(CXX) $< $(CXXFLAGS) -c -o $@

# $(OBJDIR)/main.o: CycleTimer.h

# $(OBJDIR)/hostFE.o : hostFE.cu
# 	${NVCC} ${CUDA_COMPILE_FLAGS} -c hostFE.cu -o $@