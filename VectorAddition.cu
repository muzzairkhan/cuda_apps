
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h> // for getting time to seed to srand
#include <math.h> // for ceil() UP and floor() DOWN

#define A_KILO (1024)
#define VECTOR_SIZE (4 * A_KILO)

void populateInputVectors(int* a, int* b);

void printVector(int* v);

void addWithCPU(int* c, const int* a, const int* b);

bool validateResults(const int* a, const int* b);

cudaError_t addWithCuda(int* c, const int* a, const int* b);

__global__ void addKernel(int* c, const int* a, const int* b)
{
    // Get our global thread ID designed to be an array index
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Make sure we do not go out of bounds;
    // Threads allocated could be larger than array length
    if (id < VECTOR_SIZE)
        c[id] = a[id] + b[id];
}

int main()
{
    int a[VECTOR_SIZE] = { 0 };
    int b[VECTOR_SIZE] = { 0 };
    int c[VECTOR_SIZE] = { 0 };
    int resultWithCPU[VECTOR_SIZE] = { 0 };

    // For random values generation to populate input vectors.
    srand(time(NULL));

    populateInputVectors(a, b);

    // Add vectors at host CPU in loop.
    addWithCPU(resultWithCPU, a, b);

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // Match the the vector addition results from both CPU and GPU.
    if (!validateResults(c, resultWithCPU))
    {
        fprintf(stderr, "<<<< ERROR: Vector addition failed with CUDA. >>>> \n");
        return 1;
    }
    else
        printf("VECTOR ADDITION SUCCESS.\n");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

void populateInputVectors(int* a, int* b)
{
    for (int i = 0; i < VECTOR_SIZE;i++)
    {
        a[i] = rand();
        b[i] = rand();
    }
}

void printVector(int* v)
{
    printf("[");
    for (int i = 0; i < VECTOR_SIZE;i++)
    {
        printf("%i, ", v[i]);
    }
    printf("]\n");
}

void addWithCPU(int* c, const int* a, const int* b)
{
    for (int i = 0; i < VECTOR_SIZE;i++)
    {
        c[i] = a[i] + b[i];
    }
}

bool validateResults(const int* a, const int* b)
{
    for (int i = 0; i < VECTOR_SIZE;i++)
    {
        if (a[i] != b[i])
            return false;
    }
    return true;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;
    int threadsPerBlock = 256;
    /* if the VECTOR_SIZE is less than threadsPerBlock, make sure there is atleast 1 block. */
    int blocksPerGrid = ceil(VECTOR_SIZE / threadsPerBlock);
    blocksPerGrid = blocksPerGrid == 0 ? 1 : blocksPerGrid;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, VECTOR_SIZE * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, VECTOR_SIZE * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, VECTOR_SIZE * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    printf("Launching kernel with GRID: %d and BLOCK: %d \n", blocksPerGrid, threadsPerBlock);
    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
    cudaStatus = cudaMemcpy(c, dev_c, VECTOR_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}



