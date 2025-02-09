# CUDA_KEPLER_DGEMM
CUDA dense DGEMM implementation, can reach 50% theoretical performance for large matrices

## Build commands on Linux
```
nvcc --compiler-options -fPIC,-O3 --shared -arch=sm_35 -O3 --use_fast_math --maxrregcount=128 -keep -Xptxas -v,-O3 dgemm_kernel.cu -o libdgemm_kernel.so
nvcc --compiler-options -fopenmp test_dgemm_kernel.cpp libdgemm_kernel.so -o test_dgemm_kernel
```
