
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define A_KILO (1024)
/* The dimensions (number of elements) of matrices in both ways, width or hight. */
#define SIZE (4 * A_KILO)

/* Populate the input matrices with random numbers between 0 and 99. */
void populateInputMatrix(int a[SIZE][SIZE], int b[SIZE][SIZE])
{
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            a[i][j] = rand() % 99;
            b[i][j] = rand() % 99;
        }
    }
}

/* For visual verification. */
void printMatrix(int m[SIZE][SIZE])
{
    printf("Matrix: \n");
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            printf("%d ", m[i][j]);
        }
        printf("\n");
    }
}

void multiplyWithCPU(int c[SIZE][SIZE], const int a[SIZE][SIZE], const int b[SIZE][SIZE])
{
    for (int i = 0; i < SIZE; i++) // Chose a row of A
    {
        for (int j = 0; j < SIZE; j++) // Chose a col of B
        {
            for (int k = 0; k < SIZE; k++) // loop over the elements of chosen row and chosen col
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }

    }
}

bool validateResults(const int a[SIZE][SIZE], const int b[SIZE][SIZE])
{
    {
        for (int i = 0; i < SIZE; i++)
        {
            for (int j = 0; j < SIZE; j++)
            {
                if (a[i][j] != b[i][j])
                    return false;
            }
        }
    }
    return true;
}

__global__ void multiplyMatricesKernel(int* c, const int* a, const int* b, size_t pitch)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // a row number, we shall use it for indexing matrix a
    int j = blockIdx.x * blockDim.x + threadIdx.x; // a column number, we shall use it for indexing matrix b
    int val = 0;

    if (i < SIZE && j < SIZE) // for the threads which don't have data to process
    {
        for (int k = 0; k < SIZE; k++)
        {
            val += a[(i * SIZE) + k] * b[j + (k * SIZE)];
        }
        c[j + (i * SIZE)] = val;
    }
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t multiplyWithCuda(int c[SIZE][SIZE], const int a[SIZE][SIZE], const int b[SIZE][SIZE])
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    size_t pitch;
    cudaError_t cudaStatus;
    dim3 threadsPerBlock(16, 16);

    /* number of of blocks in any dimension of grid. */
    int numBlocksXY = ceil(SIZE / threadsPerBlock.x);

    /* if the SIZE is less than threadsPerBlock, make sure there is atleast 1 block. */
    numBlocksXY = numBlocksXY == 0 ? 1 : numBlocksXY;
    dim3 blocksPerGrid(numBlocksXY, numBlocksXY);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaStatus = cudaMallocPitch(&dev_a, &pitch, SIZE * sizeof(int), SIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_a failed!");
        goto Error;
    }

    cudaStatus = cudaMallocPitch(&dev_b, &pitch, SIZE * sizeof(int), SIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_b failed!");
        goto Error;
    }

    cudaStatus = cudaMallocPitch(&dev_c, &pitch, SIZE * sizeof(int), SIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_c failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy2D(dev_a, pitch, a, SIZE * sizeof(int), SIZE * sizeof(int), SIZE, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy dev_a failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy2D(dev_b, pitch, b, SIZE * sizeof(int), SIZE * sizeof(int), SIZE, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy dev_b failed!");
        goto Error;
    }

    printf("Launching kernel with GRID: (%d %d) and BLOCK: (%d %d) \n", blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);
    // Launch a kernel on the GPU with one thread for each element.
    multiplyMatricesKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b, pitch);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multiplyMatricesKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy2D(c, SIZE * sizeof(int), dev_c, pitch, SIZE * sizeof(int), SIZE, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy dev_c failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

int main()
{
    int a[SIZE][SIZE] = { 0 };
    int b[SIZE][SIZE] = { 0 };
    int c[SIZE][SIZE] = { 0 };
    int resultWithCPU[SIZE][SIZE] = { 0 };

    // For random values generation to populate input vectors.
    srand((unsigned int)time(NULL));

    populateInputMatrix(a, b);
    //printMatrix(a);
    //printMatrix(b);

    // Multiply Matrices at host CPU.
    multiplyWithCPU(resultWithCPU, a, b);
    //printMatrix(resultWithCPU);

    // Multiply vectors in parallel.
    cudaError_t cudaStatus = multiplyWithCuda(c, a, b);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multiplyWithCuda failed!");
        return 1;
    }

    // Match the the vector addition results from both CPU and GPU.
    if (!validateResults(c, resultWithCPU))
    {
        fprintf(stderr, "<<<< ERROR: Matrix multiplication failed with CUDA. >>>> \n");
        return 1;
    }
    else
        printf("MATRIX MUTIPLICATION SUCCESS.\n");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}