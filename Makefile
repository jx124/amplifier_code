NVCC := /usr/local/cuda/bin/nvcc

all: amplifiers weighted_amplifiers

amplifiers: amplifiers_gpu.cu
	$(NVCC) -std=c++20 -O2 -dlto -arch=sm_86 --expt-relaxed-constexpr -Xcompiler -Wall -o amplifiers_gpu amplifiers_gpu.cu

weighted_amplifiers: weighted_amplifiers_gpu.cu
	$(NVCC) -std=c++20 -O2 -dlto -arch=sm_86 --expt-relaxed-constexpr -Xcompiler -Wall -o weighted_amplifiers_gpu weighted_amplifiers_gpu.cu
