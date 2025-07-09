#include <iostream>
#include <cuda_runtime.h>

#define SIZE (1024 * 1024)
#define THREADS_PER_BLOCK 256
#define BLOCKS ((SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)

__global__ void reduceKernel(int* input, int* output) {
    __shared__ int shared[THREADS_PER_BLOCK];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    shared[tid] = (idx < SIZE) ? input[idx] : 0;

    __syncthreads();

    // Redukcja drzewa: dziel przez 2, aż zostanie 1 wartość
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // Pierwszy wątek w bloku zapisuje wynik
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

int main() {
    int* h_input = new int[SIZE];
    int* h_partialSums = new int[BLOCKS];

    for (int i = 0; i < SIZE; ++i) {
        h_input[i] = 1; 
    }

    int* d_input, * d_partialSums;
    cudaMalloc(&d_input, SIZE * sizeof(int));
    cudaMalloc(&d_partialSums, BLOCKS * sizeof(int));

    cudaMemcpy(d_input, h_input, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    reduceKernel << <BLOCKS, THREADS_PER_BLOCK >> > (d_input, d_partialSums);

    cudaMemcpy(h_partialSums, d_partialSums, BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

    long long total = 0;
    for (int i = 0; i < BLOCKS; ++i) {
        total += h_partialSums[i];
    }

    std::cout << "Suma: " << total << std::endl;

    cudaFree(d_input);
    cudaFree(d_partialSums);
    delete[] h_input;
    delete[] h_partialSums;

    return 0;
}
