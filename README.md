# cuda_apps
A basic set of CUDA applications written in C language, compiled with CUDA runtime V12.6

# Matrix Multiplication
  - An example to multiply two matrices of sizes 4096 x 4096 integer entries each. The size can be changed by setting V_SIZE define which applies to both dimensions (width, hight) of the input matrices.
  - Currently chosen block size is 16x16 which results in 256 threads per block.
  - Each thread computes a single entry of the resulting matrix.
  - The input matrices are populated with random values in range of 0 to 99.
  - It utilizes cudaMallocPitch for device memory allocation and cudaMemcpy2D data transfer.
  - It verifies the results of multiplication with CUDA device by computing and comparing the results at host.

# Vector Addition
- An example to add two vectors of size 4096 integer entries each. The size can be changed by setting VECTOR_SIZE define.
  - Currently chosen block size is 16x16 which results in 256 threads per block.
  - Each thread computes a single entry of the resulting matrix.
  - The input vectors are populated with random integer values.
  - It utilizes cudaMalloc for device memory allocation and cudaMemcpy data transfer.
  - It verifies the results of addition operation performed at CUDA device by computing and comparing the results at host.
