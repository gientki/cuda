#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <stdio.h>

#define N 100000000
#define LEARNING_RATE 0.00000000001f
#define EPCHOS 50

using namespace std;

void generateData(vector<float>& x, vector<float>& y)
{
    float a = 2.0f,b = 1.0f;
    
    for (int i = 0; i < N; i++)
    {
        x[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f; //los x [0-10]
        float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.5f; //daj kurwie szumu
        y[i] = a * x[i] + b + noise;
    }
}
//CPU
void computerGradientsCPU(
    const vector<float>& x,
    const vector<float>& y,
    float a, float b,
    float& grad_a, float& grad_b){
    
    grad_a = 0.0f;
    grad_b = 0.0f;

    for (int i = 0; i < N; i++)
    {
        float y_pred = a * x[i] + b;
        float error = y_pred - y[i];

        grad_a += error * x[i];
        grad_b += error; 
    }

    grad_a /= N;
    grad_b /= N;
}
void linearRegressionCPU(const vector<float>& x, const vector<float>& y)
{
    float a = 0.0f, b = 0.0f;
    cout << "\n--- CPU TRENOWANKO --- \n";

    auto start = chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < EPCHOS; epoch++)
    {
        float grad_a, grad_b;
        computerGradientsCPU(x, y, a, b, grad_a, grad_b);

        a -= LEARNING_RATE * grad_a;
        b -= LEARNING_RATE * grad_b;
    }

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> duration = end - start;

    cout << "CPU wynik: a = " << a << " b = " << b << "\n"<<"Czas: "<<duration.count()<<"s\n";
}
//GPU
//__global__ void computeGradientsGPU(
//    float* x,float* y,
//    float a,float b,
//    float* grad_a, float* grad_b,
//    int n)  {
//    
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if (i < n)
//    {
//        float y_pred = a * x[i] + b;
//        float error = y_pred - y[i];
//        grad_a[i] = error * x[i];
//        grad_b[i] = error;
//    }
//}

// Kernel: sumy partial gradientów per-block
inline void cudaCheck(cudaError_t code, const char* file = __FILE__, int line = __LINE__) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(code));
        exit(code);
    }
}
#define cudaCheck(x) cudaCheck((x), __FILE__, __LINE__)

__global__ void computeGradientsGPU(
    const float* x, const float* y,
    float a, float b,
    float* grad_a_partial, float* grad_b_partial,
    int n)
{
    extern __shared__ float shared[];            // 2 * blockDim.x floats
    float* s_grad_a = shared;                    // [0 .. blockDim.x-1]
    float* s_grad_b = shared + blockDim.x;       // [blockDim.x .. 2*blockDim.x-1]

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (i < n) {
        float y_pred = a * x[i] + b;
        float error = y_pred - y[i];

        s_grad_a[tid] = error * x[i];
        s_grad_b[tid] = error;
    }


    __syncthreads();
    //dodawanko wszystkich s_gradow w bloku po drzewie 
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_grad_a[tid] += s_grad_a[tid + stride];
            s_grad_b[tid] += s_grad_b[tid + stride];
        }
        __syncthreads();
    }
    //pierwszy wątek w bloku wywala zapisuje liczbe gradów do globala 
    if (tid == 0) {
        grad_a_partial[blockIdx.x] = s_grad_a[0];
        grad_b_partial[blockIdx.x] = s_grad_b[0];
    }
}


void linearRegressionGPU(const std::vector<float>& x_host, const std::vector<float>& y_host, int blockSize) {
    int gridSize = (N + blockSize - 1) / blockSize;

    float* x_dev = nullptr, * y_dev = nullptr;
    float* grad_a_dev = nullptr, * grad_b_dev = nullptr;
    float a = 0.0f, b = 0.0f;

    // host buffers for partial sums (one entry per block)
    float* grad_a_host = new float[gridSize];
    float* grad_b_host = new float[gridSize];

    cudaCheck(cudaMalloc(&x_dev, N * sizeof(float)));
    cudaCheck(cudaMalloc(&y_dev, N * sizeof(float)));

    cudaCheck(cudaMalloc(&grad_a_dev, gridSize * sizeof(float)));
    cudaCheck(cudaMalloc(&grad_b_dev, gridSize * sizeof(float)));

    // Copy inputs once
    cudaCheck(cudaMemcpy(x_dev, x_host.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(y_dev, y_host.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    cout << "\n--- GPU TRENOWANKO (blockSize=" << blockSize << ", gridSize=" << gridSize << ") ---\n";
    auto start = chrono::high_resolution_clock::now();

    size_t sharedMemSize = 2 * blockSize * sizeof(float);

    for (int epoch = 0; epoch < EPCHOS; ++epoch) {
        cudaCheck(cudaMemset(grad_a_dev, 0, gridSize * sizeof(float)));
        cudaCheck(cudaMemset(grad_b_dev, 0, gridSize * sizeof(float)));

        computeGradientsGPU << <gridSize, blockSize, sharedMemSize >> > (
            x_dev, y_dev, a, b, grad_a_dev, grad_b_dev, N);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
            break;
        }

        cudaCheck(cudaDeviceSynchronize());

        // copy back only gridSize elements (partial sums)
        cudaCheck(cudaMemcpy(grad_a_host, grad_a_dev, gridSize * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(grad_b_host, grad_b_dev, gridSize * sizeof(float), cudaMemcpyDeviceToHost));

        // DEBUG: kilka partial sums
        #if 0
        if (epoch % 10 == 0) {
            cout << "partial grad_a (first 8): ";
            for (int i = 0; i < min(gridSize, 8); ++i) cout << grad_a_host[i] << " ";
            cout << "\n";
        }
        #endif

        // sumowanie partials na hoście
        double grad_a_total = 0.0;
        double grad_b_total = 0.0;
        for (int i = 0; i < gridSize; ++i) {
            grad_a_total += static_cast<double>(grad_a_host[i]);
            grad_b_total += static_cast<double>(grad_b_host[i]);
        }

        grad_a_total /= (double)N;
        grad_b_total /= (double)N;

        // update
        a -= LEARNING_RATE * static_cast<float>(grad_a_total);
        b -= LEARNING_RATE * static_cast<float>(grad_b_total);
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "wynik GPU: a = " << a << " b = " << b << "\nczas: " << duration.count() << " s\n";

    // cleanup
    cudaFree(x_dev); cudaFree(y_dev);
    cudaFree(grad_a_dev); cudaFree(grad_b_dev);
    delete[] grad_a_host; delete[] grad_b_host;
}

int main()
{
    srand(42);
    
    vector<float> x(N), y(N);
    
    generateData(x, y);
    linearRegressionCPU(x, y);

    for(int i=0;i<16;i++)
        linearRegressionGPU(x, y,pow(2,i));

    return 0; 
}
