#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <stdio.h>

#define N 100000000

using namespace std;

void generateData(vector<float>& x, vector<float>& y)
{
    float a = 2.0f, b = 1.0f;

    for (int i = 0; i < N; i++)
    {
        x[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f; //los x [0-10]
        float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.5f; //daj kurwie szumu
        y[i] = a * x[i] + b + noise;
    }
}
//CPU
void computerSmallestSquaresCPU(
    const vector<float>& x,
    const vector<float>& y,
    float& grad_a, float& grad_b) {

    double temp_x = 0;
    double temp_y = 0;
    double xy = 0;
    double xsq = 0;

    grad_a = 0.0f;
    grad_b = 0.0f;

    for (int i = 0; i < N; i++)
    {
        temp_x += x[i];
        temp_y += y[i];

        xy += x[i] * y[i];
        xsq += x[i] * x[i];
    }

    temp_x /= N;
    temp_y /= N;

    xy /= N;
    xsq /= N;

#if 1
    cout << "\ntemp_x: " << temp_x << "\ntemp_y: "
        << temp_y << "\nxy: " << xy << "\nxsq: " << xsq << endl;
#endif

    grad_a = (xy - (temp_x * temp_y)) / (xsq - temp_x * temp_x);
    grad_b = temp_y - grad_a * temp_x;
}
void linearRegressionCPU(const vector<float>& x, const vector<float>& y)
{
    float a = 0.0f, b = 0.0f;
    cout << "\n--- CPU TRENOWANKO --- \n";

    auto start = chrono::high_resolution_clock::now();
    
    float grad_a, grad_b;
    computerSmallestSquaresCPU(x, y, grad_a, grad_b);

    a =  grad_a;
    b =  grad_b;
    
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> duration = end - start;

    cout << "CPU wynik: a = " << a << " b = " << b << "\n" << "Czas: " << duration.count() << "s\n";
}
inline void cudaCheck(cudaError_t code, const char* file = __FILE__, int line = __LINE__) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(code));
        exit(code);
    }
}
#define cudaCheck(x) cudaCheck((x), __FILE__, __LINE__)

#if 0
    __global__ void computeSmallestSquaresGPU(
        float* x, float* y,
        float a, float b,
        int n,
        double* temp_x,
        double* temp_y,
        double* xy,
        double* xsq
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int tid = threadIdx.x;

        if (i < n) {
            double xi = (double)x[i];
            double yi = (double)y[i];

            atomicAdd(temp_x, xi);
            atomicAdd(temp_y, yi);
            atomicAdd(xy, xi * yi);
            atomicAdd(xsq, xi * xi);
        }

    }
#endif

    __global__ void computeSmallestSquaresGPU(
        const float* x, const float* y,
        int n,
        double* temp_x,
        double* temp_y,
        double* xy,
        double* xsq)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;  int tid = threadIdx.x;

        extern __shared__ double shared[];
        double* s_x = shared;   double* s_y = shared + blockDim.x;
        double* s_xy = shared + 2 * blockDim.x; double* s_xsq = shared + 3 * blockDim.x;

        if (i < n) {
            double xi = x[i]; double yi = y[i];

            s_x[tid] = xi; s_y[tid] = yi;
            s_xy[tid] = xi * yi;  s_xsq[tid] = xi * xi;
        }
        else {
            s_x[tid] = 0;   s_y[tid] = 0;   s_xy[tid] = 0;  s_xsq[tid] = 0;
        }

        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride) {
                s_x[tid] += s_x[tid + stride];  s_y[tid] += s_y[tid + stride];
                s_xy[tid] += s_xy[tid + stride];    s_xsq[tid] += s_xsq[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            temp_x[blockIdx.x] = s_x[0];    temp_y[blockIdx.x] = s_y[0];
            xy[blockIdx.x] = s_xy[0];   xsq[blockIdx.x] = s_xsq[0];
        }
    }



void linearRegressionGPU(const std::vector<float>& x_host, const std::vector<float>& y_host, int blockSize) {
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t totalGlobal = prop.totalGlobalMem;   size_t sharedPerBlock = prop.sharedMemPerBlock;
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;

    size_t bytes_x = (size_t)N * sizeof(float); size_t bytes_y = (size_t)N * sizeof(float);
    size_t bytes_partials = (size_t)gridSize * sizeof(double) * 4; 
    size_t estimated = bytes_x + bytes_y + bytes_partials;

    float* x_dev = nullptr, * y_dev = nullptr;
    float a = 0.0f, b = 0.0f;
    
    double *temp_x = nullptr;   double *temp_y = nullptr;
    double *xy = nullptr;   double *xsq = nullptr;

    double *h_temp_x = new double[gridSize];    double *h_temp_y = new double[gridSize];
    double *h_xy = new double[gridSize];    double *h_xsq = new double[gridSize];

    double tx = 0;  double ty = 0;
    double hxy = 0; double hxsq = 0;

    // host buffers for partial sums (one entry per block)
    float* grad_a_host = new float[gridSize];   float* grad_b_host = new float[gridSize];

    cudaCheck(cudaMalloc(&x_dev, N * sizeof(float)));   cudaCheck(cudaMalloc(&y_dev, N * sizeof(float)));

    cudaCheck(cudaMalloc(&temp_x, gridSize * sizeof(double)));  
    cudaCheck(cudaMalloc(&temp_y, gridSize * sizeof(double)));
    cudaCheck(cudaMalloc(&xy, gridSize * sizeof(double)));
    cudaCheck(cudaMalloc(&xsq, gridSize * sizeof(double)));

    // Copy inputs once
    cudaCheck(cudaMemcpy(x_dev, x_host.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(y_dev, y_host.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    cudaCheck(cudaMemset(temp_x, 0, gridSize * sizeof(double)));
    cudaCheck(cudaMemset(temp_y, 0, gridSize * sizeof(double)));

    cudaCheck(cudaMemset(xy, 0, gridSize * sizeof(double)));
    cudaCheck(cudaMemset(xsq, 0, gridSize * sizeof(double)));

    cout << "\n--- GPU TRENOWANKO (blockSize=" << blockSize << ", gridSize=" << gridSize << ") ---\n";
    auto start = chrono::high_resolution_clock::now();

    size_t sharedMemSize = 4 * blockSize * sizeof(double);
    

    computeSmallestSquaresGPU << <gridSize, blockSize, sharedMemSize >> > (
            x_dev, y_dev, N,
            temp_x,temp_y,xy,xsq);

    cudaError_t err = cudaGetLastError();
        
    if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        }

    cudaCheck(cudaDeviceSynchronize());

        // copy back only gridSize elements (partial sums)

        cudaCheck(cudaMemcpy(h_temp_x, temp_x, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_temp_y, temp_y, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_xy, xy, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_xsq, xsq, gridSize * sizeof(double), cudaMemcpyDeviceToHost));

        for (int i = 0; i < gridSize; i++)
        {
            tx += h_temp_x[i];
            ty += h_temp_y[i];
            hxy += h_xy[i];
            hxsq += h_xsq[i];
        }

        tx /= N;
        ty /= N;

        hxy /= N;
        hxsq /= N;

        a = (hxy - (tx * ty)) / (hxsq - tx * tx);
        b = ty - a * tx;
#if 1
        cout << "\ntemp_x: " << tx << "\ntemp_y: "
            << ty << "\nxy: " << hxy << "\nxsq: " << hxsq << endl;
        // DEBUG: kilka partial sums
#endif
#if 0

        if (epoch % 10 == 0) {
            cout << "partial grad_a (first 8): ";
            for (int i = 0; i < min(gridSize, 8); ++i) cout << grad_a_host[i] << " ";
            cout << "\n";
        }
#endif

   

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "Device totalGlobalMem = " << totalGlobal / 1024 / 1024 << " MB\n";
    cout << "gridSize=" << gridSize << " blockSize=" << blockSize
        << " sharedPerBlock=" << sharedPerBlock << " maxThreadsPerBlock=" << maxThreadsPerBlock << "\n";
    cout << "Estimated memory (x+y+partials) = " << estimated / 1024 / 1024 << " MB\n";

    cout << "wynik GPU: a = " << a << " b = " << b << "\nczas: " << duration.count() << " s\n";


    cudaFree(x_dev);    cudaFree(y_dev);
    cudaFree(temp_x);   cudaFree(temp_y);
    cudaFree(xy);   cudaFree(xsq);

    delete[] grad_a_host;   delete[] grad_b_host;
    delete[] h_temp_x;  delete[] h_temp_y;
    delete[] h_xy;   delete[] h_xsq;
}

int main()
{
    srand(42);

    vector<float> x(N), y(N);

    generateData(x, y);
    linearRegressionCPU(x, y);

    for (int i = 2; i < 12; i++)
        linearRegressionGPU(x, y, pow(2, i));

    return 0;
}
