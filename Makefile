NVCC := /usr/local/cuda/bin/nvcc

all: cuda

cuda: amplifiers_gpu.cu
	$(NVCC) -std=c++20 -O2 -dlto -arch=sm_86 -ftz=true --expt-relaxed-constexpr -Xcompiler -Wall -o amplifiers_gpu amplifiers_gpu.cu
