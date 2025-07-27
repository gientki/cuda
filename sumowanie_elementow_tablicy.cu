#include <iostream>
#include <cuda_runtime.h>

#define SIZE (1024 * 1024)
#define THREADS_PER_BLOCK 256
#define BLOCKS ((SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) //4096


#define FINAL_BLOCKS ((BLOCKS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) //16

__global__ void reduceKernel(int* input, int* output, int n) {
    __shared__ int shared[THREADS_PER_BLOCK];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Załaduj dane do shared memory
    shared[tid] = (idx < n) ? input[idx] : 0;

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

    // Inicjalizacja danych
    for (int i = 0; i < SIZE; ++i) {
        h_input[i] = 1; // np. same 1-ki, oczekiwany wynik = SIZE
    }

    int* d_input, * d_partialSums;
    cudaMalloc(&d_input, SIZE * sizeof(int));
    cudaMalloc(&d_partialSums, BLOCKS * sizeof(int));

    cudaMemcpy(d_input, h_input, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Uruchom redukcję
    //metoda 1
    reduceKernel << <BLOCKS, THREADS_PER_BLOCK >> > (d_input, d_partialSums, SIZE);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(h_partialSums, d_partialSums, BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

    // Suma końcowa na CPU
    long long total = 0;
    for (int i = 0; i < BLOCKS; ++i) {
        total += h_partialSums[i];
    }

    std::cout << "Suma: " << total << std::endl;
    //metoda 2
    //Zbierz wyniki częściowe
    std::cout << "Suma (z 0 etapu GPU): " << h_partialSums[0] << " po: " << (int)BLOCKS << " blokow" << std::endl;
    int* d_finalSum;
    cudaMalloc(&d_finalSum, FINAL_BLOCKS * sizeof(int));
    
    reduceKernel << <FINAL_BLOCKS, THREADS_PER_BLOCK >> > (d_partialSums, d_finalSum, BLOCKS);
    
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    int result = 0;
    cudaMemcpy(&result, d_finalSum, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Suma (z 1 etapu GPU): " << result<< " po: "<< (int)FINAL_BLOCKS << " blokow" << std::endl;
    int* d_finalResult;
    cudaMalloc(&d_finalResult, sizeof(int));

    reduceKernel << <1, THREADS_PER_BLOCK >> > (d_finalSum, d_finalResult, FINAL_BLOCKS);

     result = 0;
    cudaMemcpy(&result, d_finalResult, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Suma (z 2 etapu GPU): " << result << std::endl;

    cudaFree(d_finalResult);
    

    cudaFree(d_input);
    cudaFree(d_partialSums);
    cudaFree(d_finalSum);

    delete[] h_input;
    delete[] h_partialSums;

    return 0;
}
