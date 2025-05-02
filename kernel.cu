#include<iostream>
#include<cstdlib>
#include<cuda_runtime.h>

using namespace std;

__global__ void addKernel(int* a, int* b, int* c, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int temp[3];

	if(idx < size)
	{
		temp[threadIdx.x] = a[idx] + b[idx];
		__syncthreads();

		c[idx] = temp[threadIdx.x];
		/*c[idx] = threadIdx.x;*/
	}
}
int main()
{
	const int size = 12;
	const int blocks = 4;

	int h_a[size] = {1,4,6,2,3,7,9,6,2,1,6,6};
	int h_b[size] = {0,1,2,3,4,5,6,7,8,9,0,1};
	int h_c[size];
	
	/*for (int i = 0; i < size; i++)
	{
		h_a[i] = rand() % 10;
		h_b[i] = rand() % 10;
	}*/
	
	int* d_a, * d_b, * d_c;
	
	
	cudaMalloc((void**)&d_a, size * sizeof(int));
	cudaMalloc((void**)&d_b, size * sizeof(int));
	cudaMalloc((void**)&d_c, size * sizeof(int));
	/*cout << d_a << endl;*/
	cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);
	
	addKernel<<<blocks, size/blocks>>>(d_a, d_b, d_c, size);

	cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < size; i++)
		cout << h_c[i] << endl;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
		return 0;
}