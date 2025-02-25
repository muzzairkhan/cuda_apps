# Abstractions in CUDA hierarchical execution model:

These abstractions provide multiple advantages while solving challenging problems.
- A natural way of realizing the parallel processing aligned with various kind of data sets like 1D (Vectors), 2D (Images, Matrices) and 3D (Volumes, Space).
- Accessing data (or memory) stored together (adjacent) by a set of processors. - Memory coalescing.     
- Special provision of synchronizing processing elements (threads and cores) within a hierarchical level.
  
## 1. Threads
An execution unit running in bulk quantity, in parallel over a large number of GPU cores. All threads have same code (instructions) but different data to be processed in Single Instruction Multiple Data (SIMD) fashion.

## 2. Blocks
Different GPUs have different number of Streaming Multiprocessors (SMs) who have the capability of executing multiple threads in parallel with some (small) amount of memory per SM shared between the threads being executed.
This shared memory is way faster than the global memory of GPU.

Blocks are simply the collection of threads. The CUDA hardware doesn't execute threads individually but in groups of threads called blocks. A block is always executed on an assigned SM. There is a limit to the number of threads per block (1024), since all threads of a block are expected to reside on the same streaming multiprocessor core and must share the limited memory resources of that core.

Greater the number of SMs in a GPU, greater the number of blocks being executed in parallel. So blocks allow CUDA to distribute work efficiently across any GPU. 

Blocks can be defined in various dimensions 1D, 2D or 3D. This provides a natural way to invoke computation across the elements in a domain such as a vector, matrix, or volume.
 
## 3. Grids
Blocks are organized into a one-dimensional, two-dimensional, or three-dimensional grid of thread blocks. The number of thread blocks in a grid is usually dictated by the size of the data being processed, which typically exceeds the number of processors in the system.

### Why multi dimensional Grids?
A **3D grid in CUDA** does not directly map to the physical architecture of the GPU, but it **optimally distributes work** across the Streaming Multiprocessors (SMs) and CUDA cores, leading to better resource utilization.
A 3D grid organizes memory access in a way that **allows coalesced memory access**. A 3D grid enables **structured workloads** (e.g., volumetric data, physics simulations) to be processed without complex index calculations.

Example: **3D Computational Fluid Dynamics (CFD)**
- The fluid simulation space is divided into **cubic sub-regions** (blocks).
- Each block is processed by a different SM.

## 4. Warps
Threads inside a block are executed in groups of 32 threads, called warps. This is due to the SIMD hardware architecture.
A block with 128 threads will be split into 4 warps (128 / 32 = 4 warps).
Each warp is scheduled independently, ensuring maximum parallel execution. Since this abstraction is FIXED and not editable, it is not made part of programming model. But knowing this fact may impact optimally defining the sizes of blocks and grid to the scale of problem or data set.

# The perspective of concurrency in CUDA
Within a block, there are warps of 32 threads each. The CUDA hardware schedules warps for execution.
Warps from different blocks (affined to an SM) can be scheduled for an SM concurrently.  
Each SM can handle multiple blocks concurrently, but the exact number depends on the specific GPU architecture and the resource requirements of the blocks.

# CUDA Programming Model

## 1. kernel 
A C/C++ like function, when called, is executed N times in parallel by N different CUDA threads. It carries the code representing a CUDA thread and specifies the associated Grid and Blocks additionally. A kernel is executed by multiple equally-shaped thread blocks.

Each thread that executes the kernel is given a unique thread ID, accessible within the kernel through built-in variables. **threadIdx** is a 3-component vector, to identify 1D, 2D or 3D thread index, forming a 1D, 2D or 3D block of threads.

Each block within the grid can be identified by a 1D, 2D or 3D unique index accessible within the kernel through the built-in **blockIdx** variable. The
dimension of the thread block is accessible within the kernel through the built-in **blockDim** variable.

## 2. Synchronization
Threads within a block can cooperate by sharing data through some shared memory and by synchronizing their execution to coordinate memory accesses. More precisely, one can specify synchronization points in the kernel by calling the __syncthreads() intrinsic function; which acts as a barrier at which all threads in the block must wait before any one is allowed to proceed.

## 3. Streams
Sequence of operations (such as kernel launches, memory copies, etc.) that execute in order on the GPU. We define streams to manage and overlap computation and data transfers, enabling better utilization of the GPU's resources. 
- Tasks in different streams can run **concurrently**, provided there are no dependencies between them. 
- Explicit streams allow controlling the execution order. 
- If no stream is specified, all operations are executed in the default stream.
- We can synchronize the host with a specific stream using **cudaStreamSynchronize**, which ensures that all operations in the stream are completed before the host continues.
- Alternatively, **cudaDeviceSynchronize** synchronizes the host with all the streams on the device.
- Use **cudaMemcpyAsync** and explicit streams to overlap computation and data transfers.

## 4. Grid Stride loops
 
 A technique used to efficiently process large datasets that exceed the number of available GPU threads. The kernel loops over the data array one grid-size at a time. The stride of the loop is *blockDim.x* X *gridDim.x* which is the total number of threads in the grid. So if there are 1280 threads in the grid, thread 0 will compute elements 0, 1280, 2560, etc. This is why it is called a grid-stride loop.
 https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/

# CUDA Memory Model

 ## 1. Memory Types
 CUDA provides several types of memories that can be used to optimize the performance of applications running on NVIDIA GPUs. Here are the main types of memory in CUDA:

1. **Global Memory**: This is the largest memory space on the GPU and is accessible by all threads in all blocks. It has high latency but high bandwidth. Data can be transferred between the host (CPU) and the global memory on the device (GPU).

2. **Shared Memory**: This is a block-level memory that is shared among all threads within the same block. It is much faster than global memory and is used for data that needs to be accessed frequently by multiple threads within a block.

3. **Local Memory**: This is a per-thread memory space that is used for **register spills**, private data that doesn't fit in the thread's registers, and stack frames. It is actually allocated in global memory and thus has high latency.

4. **Constant Memory**: This is a cached memory space that is read-only for the kernel. It is optimized for broadcasting the same value to all threads in a warp and has lower latency than global memory when all threads access the same location.

5. **Texture Memory**: This is a specialized memory that is optimized for 2D spatial locality. It is read-only and cached, making it efficient for applications that require texture sampling or where memory access patterns exhibit spatial locality.

6. **Registers**: These are the fastest memory available to each thread. Variables that are declared within a kernel without any specific memory qualifiers are typically stored in registers. However, the number of registers is limited, and if a kernel uses too many registers, some variables may be spilled to local memory.

7. **Pinned (Page-Locked) Memory**: This is a type of host memory that is page-locked and can be accessed directly by the GPU, allowing for faster data transfers between the host and the device.

8. **Unified Memory**: Introduced in CUDA 6, unified memory creates a pool of managed memory that is shared between the CPU and GPU, bridging the CPU-GPU divide. The system automatically migrates data between the host and device as needed, simplifying memory management.

Each type of memory has its own use case and performance characteristics, and efficient CUDA programming often involves carefully managing data placement and movement between these different memory types to optimize performance.
 
 ## 2. Pitched Memory allocation and copying
 Memory access on the GPU works much better if the data items are aligned. Hence, allocating 2D or 3D arrays so that every row starts at a 64-byte or 128-byte boundary address will improve performance, for which CUDA offers special memory allocation and memory copy APIs; **cudaMallocPitch()**, **cudaMemcpy2D()** etc. Each row may be padded for alignment. The drawbacks are some wasted space and a bit more complicated elements access.
 https://nichijou.co/cudaRandom-memAlign/
 
 ## 3. Memory Coalescing 
 - This technique takes advantage of the fact that threads in a warp execute the same instruction at any given point in time.
 - The most favourable access pattern is achieved when all threads in a warp access **consecutive global memory locations**.
 - When all threads in a warp execute a load instruction, the hardware detects whether they access consecutive global memory locations. If that’s the case, the hardware combines (coalesces) all these accesses into a consolidated access to consecutive DRAM locations. Such coalesced access allows the DRAMs to **deliver data as a burst**.
 https://nichijou.co/cuda5-coalesce/
 

 

